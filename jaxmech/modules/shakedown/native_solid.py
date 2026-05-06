"""Native solid CVXPY shakedown orchestration from MAT-first inputs."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from jaxmech.modules.shakedown.postprocess import save_shakedown_summary
from jaxmech.modules.shakedown.preprocess import (
    build_parameter_sets,
    build_solid_fem_input,
    load_case_mats,
    resolve_case_io,
)
from jaxmech.modules.shakedown.result import ShakedownResult
from jaxmech.optimize.cvxpy import build_lower_bound_problem, solve_version_c


def _result_bundle_name(config: dict) -> str:
    return f"VersionC_{int(config['num_vert'])}P.mat"


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
    return 0.0 if arr.size == 0 else float(np.sum(np.abs(arr)))


def _yield_violation_norm1(yield_ratio) -> float:
    arr = np.asarray(yield_ratio, dtype=np.float64)
    return 0.0 if arr.size == 0 else float(np.sum(np.maximum(arr - 1.0, 0.0)))


def _build_elastic_input_set(case_io: list[tuple[str, Path, Path]], fem_input: dict) -> dict:
    sigma_cases = [np.asarray(fem_input["sigma_E"], dtype=np.float64)]
    if "sigma_E_cases" in fem_input:
        extra = np.asarray(fem_input["sigma_E_cases"], dtype=np.float64)
        sigma_cases.extend([extra[:, :, i] for i in range(extra.shape[2])])
    elastic_stress = np.stack(sigma_cases, axis=2)
    return {
        "C_sparse": fem_input["C_sparse"].tocsr(),
        "ele_yield": np.asarray(fem_input["ele_yield"], dtype=np.float64).reshape(-1, 1),
        "elem_types": _object_column(np.asarray(fem_input["elem_types"], dtype=object).reshape(-1)),
        "free_dofs": np.asarray(fem_input["free_dofs"], dtype=np.int64).reshape(1, -1),
        "gauss_coords": np.asarray(fem_input["gauss_coords"], dtype=np.float64),
        "gauss_vols": np.asarray(fem_input["gauss_vols"], dtype=np.float64).reshape(-1, 1),
        "n_str": np.asarray([[int(fem_input["n_str"])]], dtype=np.int32),
        "gauss_stress_cases": np.asarray(elastic_stress, dtype=np.float64),
        "mat_files": _object_column([str(item[2]) for item in case_io]),
    }


def _build_config_info(config: dict) -> dict:
    mat_paths = _object_column(config["mat_files"])
    load_factor_set = (
        np.asarray(config["load_factor_set"], dtype=np.float64)
        if config["load_factor_set"] is not None
        else np.zeros((0, 3), dtype=np.float64)
    )
    return {
        "family": np.asarray(["solid"]),
        "formulation": np.asarray(["C"]),
        "solver_backend": np.asarray(["cvxpy"]),
        "mat_files": mat_paths,
        "R_ratios": np.asarray(config["R_ratios"], dtype=np.float64).reshape(1, -1),
        "use_angles": np.asarray([[int(bool(config["use_angles"]))]], dtype=np.int32),
        "theta_deg": np.asarray(config["theta_deg"], dtype=np.float64).reshape(1, -1),
        "phi_deg": np.asarray(config["phi_deg"], dtype=np.float64).reshape(1, -1),
        "load_factor_set": load_factor_set,
        "NumVert": np.asarray([[int(config["num_vert"])]], dtype=np.int32),
        "yield_values": np.asarray([repr(config["yield_values"])]),
    }


def _solve_one(config: dict, fem_input: dict, params: dict):
    problem = build_lower_bound_problem(
        fem_input=fem_input,
        independent_case_ids=np.arange(len(config["load_cases"]), dtype=np.int32),
        n_vertices=int(config["num_vert"]),
        theta_deg=None if len(config["load_cases"]) == 1 else params["theta_deg"],
        phi_deg=None if len(config["load_cases"]) <= 2 else params["phi_deg"],
        load_scale=params["load_scale"],
        R_ratios=config["R_ratios"],
    )
    solution = solve_version_c(problem, solver="CLARABEL")
    return problem, solution


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
        "equilibrium_residual_norm1": np.asarray([[_norm1(solution.equilibrium_residual)]], dtype=np.float64),
        "yield_violation_norm1": np.asarray([[_yield_violation_norm1(solution.yield_ratio)]], dtype=np.float64),
        "active_gauss_count": np.asarray([[float(solution.active_gauss_count)]], dtype=np.float64),
        "active_gauss_fraction": np.asarray([[float(solution.active_gauss_fraction)]], dtype=np.float64),
    }


def _build_summary_result(results: list[dict], config_path: str | Path, config: dict, summary_path: Path) -> ShakedownResult:
    alphas = [float(np.asarray(item["objective_value"]).reshape(-1)[0]) for item in results]
    return ShakedownResult(
        alpha=max(alphas),
        status="optimal",
        solver="VersionC",
        summary_mat_path=str(summary_path),
        residual_stress=np.asarray(results[0]["residual_stress"], dtype=np.float64),
        yield_ratio=np.asarray(results[0]["yield_ratio"], dtype=np.float64),
        metadata={"n_parameter_sets": len(results), "all_alphas": alphas, "config_path": str(config_path)},
    )


def run_native_solid(config_path: str | Path, config: dict, *, step: str = "all") -> ShakedownResult:
    """Run solid shakedown natively from JAX MAT files using CVXPY."""
    if step != "all":
        raise ValueError(f"Solid CVXPY solver only supports step='all', got {step!r}.")

    print("[SD] Loading solid elastic MAT files...")
    case_io = resolve_case_io(config)
    case_raw = load_case_mats(case_io)
    print("[SD] Building FEM input...")
    fem_input = build_solid_fem_input(case_raw, config["yield_values"])
    parameter_sets = build_parameter_sets(config, case_io, case_raw)

    print(
        f"[SD] Cases={len(case_io)}, gauss={fem_input['n_gauss']}, "
        f"components={fem_input['n_str']}, parameter_sets={len(parameter_sets)}"
    )

    result_dir = Path(str(config.get("workflow_dir", case_io[0][2].parent))).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, params in enumerate(parameter_sets):
        print(f"[SD] Solving parameter set {i + 1}/{len(parameter_sets)}...")
        problem, solution = _solve_one(config, fem_input, params)
        alpha_i = float(np.asarray(solution.objective_value).reshape(-1)[0])
        print(f"[SD]   alpha = {alpha_i:.6f}, status = {solution.status}, time = {solution.solve_time:.2f}s")
        results.append(_solution_record(config, problem, solution))

    summary_path = result_dir / _result_bundle_name(config)
    print(f"[SD] Saving result MAT: {summary_path}")
    save_shakedown_summary(
        summary_path,
        elastic_input=_build_elastic_input_set(case_io, fem_input),
        results=results,
        config_info=_build_config_info(config),
        metadata={"config_path": str(config_path), "created_at": dt.datetime.now().isoformat(), "family": "solid"},
    )
    print("[SD] Shakedown analysis completed.")
    return _build_summary_result(results, config_path, config, summary_path)


def run_native_solid_cvxpy(config_path: str | Path, config: dict) -> ShakedownResult:
    return run_native_solid(config_path, config, step="all")
