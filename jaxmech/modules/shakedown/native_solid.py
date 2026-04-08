"""Native solid shakedown orchestration from MAT-first inputs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from jaxmech.env.env import PROJECT_ROOT
from jaxmech.modules.shakedown.postprocess import save_shakedown_summary
from jaxmech.modules.shakedown.preprocess import (
    build_parameter_sets,
    build_solid_fem_input,
    load_case_mats,
    resolve_case_io,
)
from jaxmech.modules.shakedown.result import ShakedownResult
from jaxmech.optimize.cvxpy import (
    build_lower_bound_problem,
    solve_version_c,
)

try:
    from jaxmech.optimize.gurobi import (
        cleanup_solid_gurobi_run,
        collect_solid_gurobi_solution,
        prepare_solid_gurobi_run,
        try_solid_gurobi_interop,
    )
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False


def _result_bundle_name(config: dict) -> str:
    solver_key = str(config["solver"])
    num_vert = int(config["num_vert"])
    suffix = {
        "VersionC": "VersionC",
        "VersionC+Gurobi": "VersionC_Gurobi",
        "VersionMN": "VersionMN",
        "VersionMN+Gurobi": "VersionMN_Gurobi",
    }[solver_key]
    return f"{suffix}_{num_vert}P.mat"


def _pad_load_factor(load_scale) -> np.ndarray:
    out = np.zeros((3,), dtype=np.float64)
    vec = np.asarray(load_scale, dtype=np.float64).reshape(-1)
    out[: vec.size] = vec
    return out


def _float_vector_or_empty(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _object_column(items) -> np.ndarray:
    arr = np.empty((len(items), 1), dtype=object)
    for i, item in enumerate(items):
        arr[i, 0] = str(item)
    return arr


def _norm1(value) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sum(np.abs(arr)))


def _yield_violation_norm1(yield_ratio) -> float:
    arr = np.asarray(yield_ratio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sum(np.maximum(arr - 1.0, 0.0)))


def _build_elastic_input_set(
    case_io: list[tuple[str, Path, Path]],
    case_raw: list[dict],
    fem_input: dict,
) -> dict:
    n_cases = len(case_io)
    validated_arr = np.zeros((n_cases, 1), dtype=np.int32)
    odb_arr = np.empty((n_cases, 1), dtype=object)
    err_items = []

    for i, ((_, _, _case_mat), raw) in enumerate(zip(case_io, case_raw)):
        validated_arr[i, 0] = int(np.asarray(raw.get("validated_with_ODB", [[0]])).reshape(-1)[0])
        odb_arr[i, 0] = (
            str(np.asarray(raw.get("validated_ODBName", np.asarray([""]))).reshape(-1)[0])
            if "validated_ODBName" in raw
            else ""
        )
        err_items.append(raw.get("Error_with_ODB", np.empty((0, 0), dtype=object)))

    sigma_cases = [np.asarray(fem_input["sigma_E"], dtype=np.float64)]
    if "sigma_E_cases" in fem_input:
        extra = np.asarray(fem_input["sigma_E_cases"], dtype=np.float64)
        sigma_cases.extend([extra[:, :, i] for i in range(extra.shape[2])])
    elastic_stress = np.stack(sigma_cases, axis=2)

    return {
        "validated_with_odb": validated_arr,
        "validated_odb_name": odb_arr,
        "error_with_odb": np.asarray(err_items, dtype=object).reshape(-1, 1),
        "C_sparse": fem_input["C_sparse"].tocsr(),
        "ele_yield": np.asarray(fem_input["ele_yield"], dtype=np.float64).reshape(-1, 1),
        "elem_types": _object_column(np.asarray(fem_input["elem_types"], dtype=object).reshape(-1)),
        "free_dofs": np.asarray(fem_input["free_dofs"], dtype=np.int64).reshape(1, -1),
        "gauss_coords": np.asarray(fem_input["gauss_coords"], dtype=np.float64),
        "gauss_vols": np.asarray(fem_input["gauss_vols"], dtype=np.float64).reshape(-1, 1),
        "n_str": np.asarray([[int(fem_input["n_str"])]], dtype=np.int32),
        "gauss_stress_cases": np.asarray(elastic_stress, dtype=np.float64),
    }


def _build_config_info(config: dict) -> dict:
    mat_paths = np.empty((len(config["mat_files"]), 1), dtype=object)
    for i, mat_file in enumerate(config["mat_files"]):
        mat_paths[i, 0] = str(mat_file)
    load_factor_set = (
        np.asarray(config["load_factor_set"], dtype=np.float64)
        if config["load_factor_set"] is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    return {
        "family": np.asarray([config["family"]]),
        "formulation": np.asarray([config["formulation"]]),
        "solver_backend": np.asarray([config["solver_backend"]]),
        "gurobi_backend": np.asarray([config["gurobi_backend"]]),
        "mat_files": mat_paths,
        "R_ratios": np.asarray(config["R_ratios"], dtype=np.float64).reshape(1, -1),
        "use_angles": np.asarray([[int(bool(config["use_angles"]))]], dtype=np.int32),
        "theta_deg": np.asarray(config["theta_deg"], dtype=np.float64).reshape(1, -1),
        "phi_deg": np.asarray(config["phi_deg"], dtype=np.float64).reshape(1, -1),
        "load_factor_set": load_factor_set,
        "NumVert": np.asarray([[int(config["num_vert"])]], dtype=np.int32),
        "yield_values": np.asarray([repr(config["yield_values"])]),
        "gurobi_params": config["gurobi_params"],
    }


def _solve_one(config: dict, fem_input: dict, params: dict):
    common = {
        "fem_input": fem_input,
        "independent_case_ids": np.arange(len(config["load_cases"]), dtype=np.int32),
        "n_vertices": int(config["num_vert"]),
        "theta_deg": None if len(config["load_cases"]) == 1 else params["theta_deg"],
        "phi_deg": None if len(config["load_cases"]) <= 2 else params["phi_deg"],
        "load_scale": params["load_scale"],
        "R_ratios": config["R_ratios"],
    }
    solver_key = str(config["solver"])
    if solver_key == "VersionC":
        problem = build_lower_bound_problem(**common)
        solution = solve_version_c(problem, solver="CLARABEL")
        return problem, solution, "VersionC"
    raise ValueError(
        f"This demo only supports solver 'VersionC'. Got {solver_key!r}."
    )


def _solution_record(config: dict, problem, solution) -> dict:
    return {
        "NumVert": np.asarray([[int(config["num_vert"])]], dtype=np.int32),
        "load_factor": _pad_load_factor(problem.load_scale),
        "status": str(solution.status),
        "solve_time": np.asarray([[float(solution.solve_time)]], dtype=np.float64),
        "objective_value": np.asarray([[float(solution.objective_value)]], dtype=np.float64),
        "primal_variables": _float_vector_or_empty(solution.primal_variables),
        "dual_variables_eq": _float_vector_or_empty(solution.dual_variables_eq),
        "dual_variables_ineq": _float_vector_or_empty(solution.dual_variables_ineq),
        "residual_stress": np.asarray(solution.residual_stress, dtype=np.float64),
        "yield_ratio": np.asarray(solution.yield_ratio, dtype=np.float64),
        "equilibrium_residual_norm1": np.asarray(
            [[_norm1(solution.equilibrium_residual)]], dtype=np.float64
        ),
        "yield_violation_norm1": np.asarray(
            [[_yield_violation_norm1(solution.yield_ratio)]], dtype=np.float64
        ),
        "active_gauss_count": np.asarray([[float(solution.active_gauss_count)]], dtype=np.float64),
        "active_gauss_fraction": np.asarray([[float(solution.active_gauss_fraction)]], dtype=np.float64),
    }


def _build_summary_result(
    results: list[dict],
    config_path: str | Path,
    config: dict,
    summary_path: Path,
) -> ShakedownResult:
    return ShakedownResult(
        alpha=max(float(np.asarray(item["objective_value"]).reshape(-1)[0]) for item in results),
        status="optimal",
        solver=str(config["solver"]),
        summary_mat_path=str(summary_path),
        residual_stress=np.asarray(results[0]["residual_stress"], dtype=np.float64),
        yield_ratio=np.asarray(results[0]["yield_ratio"], dtype=np.float64),
        metadata={
            "n_parameter_sets": len(results),
            "all_alphas": [float(np.asarray(item["objective_value"]).reshape(-1)[0]) for item in results],
            "config_path": str(config_path),
        },
    )


def run_native_solid(
    config_path: str | Path,
    config: dict,
    *,
    step: str = "all",
) -> ShakedownResult:
    """Run solid shakedown natively from JAX MAT files."""
    print("[SD] 解析配置，加载 MAT 文件...")
    case_io = resolve_case_io(config)
    case_raw = load_case_mats(case_io)
    print("[SD] 构建 FEM 输入...")
    fem_input = build_solid_fem_input(case_raw, config["yield_values"])
    parameter_sets = build_parameter_sets(config, case_io, case_raw)

    n_cases = len(case_io)
    n_params = len(parameter_sets)
    solver_key = str(config["solver"])
    print(f"[SD] 加载 {n_cases} 个 MAT，{fem_input['n_gauss']} 个高斯点，{fem_input['n_str']} 应力分量")
    print(f"[SD] 求解器: {solver_key}，NVert={config['num_vert']}，参数集: {n_params} 组")

    result_dir = Path(str(config.get("workflow_dir", case_io[0][2].parent))).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    is_gurobi = solver_key.endswith("+Gurobi")

    if is_gurobi:
        raise NotImplementedError(
            "Gurobi solver is not available in this demo. "
            "Use solver = cvxpy in your shakedown config."
        )
    elif step != "all":
        raise ValueError(f"Solid CVXPY solver only supports step='all', got {step!r}.")

    results = []
    for i, params in enumerate(parameter_sets):
        print(f"[SD] 参数集 {i+1}/{n_params}: 开始求解 ...")
        problem, solution, _ = _solve_one(config, fem_input, params)
        alpha_i = float(np.asarray(solution.objective_value).reshape(-1)[0])
        print(f"[SD]   alpha = {alpha_i:.6f},  status = {solution.status},  time = {solution.solve_time:.2f}s")
        results.append(_solution_record(config, problem, solution))

    print(f"[SD] 保存结果到 {result_dir} ...")
    summary_path = result_dir / _result_bundle_name(config)

    extra_meta: dict = {"config_path": str(config_path), "created_at": dt.datetime.now().isoformat()}

    save_shakedown_summary(
        summary_path,
        elastic_input=_build_elastic_input_set(case_io, case_raw, fem_input),
        results=results,
        config_info=_build_config_info(config),
        metadata=extra_meta,
    )
    print("[SD] 安定分析完成。")
    return _build_summary_result(results, config_path, config, summary_path)



def run_native_solid_cvxpy(
    config_path: str | Path,
    config: dict,
) -> ShakedownResult:
    """Backward-compatible wrapper for the solid CVXPY path."""
    return run_native_solid(config_path, config, step="all")
