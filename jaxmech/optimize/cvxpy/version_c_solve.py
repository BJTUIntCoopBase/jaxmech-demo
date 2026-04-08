"""
SD_VersionC_solve
=================
Solve and post-process the Version-C shakedown problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import scipy.io as sio

from jaxmech.optimize.cvxpy.version_c import SDVersionCProblem


@dataclass(frozen=True)
class SDVersionCSolution:
    """Optimization result and post-processed fields for Version-C."""

    status: str
    solver_name: str
    solve_time: float
    objective_value: float | None
    alpha: float | None
    primal_variables: np.ndarray | None
    primal_variable_index: dict | None
    dual_variables_eq: np.ndarray | None
    dual_variables_ineq: np.ndarray | None
    dual_variable_index_eq: dict | None
    dual_variable_index_ineq: dict | None
    residual_stress: np.ndarray | None
    yield_ratio: np.ndarray | None
    equilibrium_residual: np.ndarray | None
    active_gauss_count: int | None
    active_gauss_fraction: float | None
    problem_summary: dict
    metadata: dict

    @property
    def solved(self) -> bool:
        return self.alpha is not None and self.primal_variables is not None


def _encode_meta_value(value):
    if isinstance(value, str):
        return np.asarray([value])
    if isinstance(value, bool):
        return np.asarray([[int(value)]], dtype=np.int64)
    if isinstance(value, dict):
        return value
    if np.isscalar(value):
        return np.asarray([[value]])
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    return arr


def _solution_payload(solution: SDVersionCSolution) -> dict:
    payload = {
        "status": np.asarray([solution.status]),
        "solver_name": np.asarray([solution.solver_name]),
        "solve_time": np.asarray([[float(solution.solve_time)]], dtype=np.float64),
        "objective_value": np.asarray(
            [[np.nan if solution.objective_value is None else float(solution.objective_value)]],
            dtype=np.float64,
        ),
        "alpha": np.asarray(
            [[np.nan if solution.alpha is None else float(solution.alpha)]],
            dtype=np.float64,
        ),
        "solved": np.asarray([[int(bool(solution.solved))]], dtype=np.int32),
    }
    for key, value in solution.problem_summary.items():
        payload[f"summary_{key}"] = _encode_meta_value(value)
    for key, value in solution.metadata.items():
        payload[f"meta_{key}"] = _encode_meta_value(value)

    array_fields = {
        "primal_variables": solution.primal_variables,
        "dual_variables_eq": solution.dual_variables_eq,
        "dual_variables_ineq": solution.dual_variables_ineq,
        "residual_stress": solution.residual_stress,
        "yield_ratio": solution.yield_ratio,
        "equilibrium_residual": solution.equilibrium_residual,
    }
    for key, value in array_fields.items():
        if value is not None:
            payload[key] = np.asarray(value, dtype=np.float64)

    if solution.primal_variable_index is not None:
        payload["primal_variable_index"] = solution.primal_variable_index
    if solution.dual_variable_index_eq is not None:
        payload["dual_variable_index_eq"] = solution.dual_variable_index_eq
    if solution.dual_variable_index_ineq is not None:
        payload["dual_variable_index_ineq"] = solution.dual_variable_index_ineq

    for key, value in {
        "active_gauss_count": solution.active_gauss_count,
        "active_gauss_fraction": solution.active_gauss_fraction,
    }.items():
        payload[key] = np.asarray(
            [[np.nan if value is None else float(value)]],
            dtype=np.float64,
        )

    return payload


def save_solution_npz(solution: SDVersionCSolution, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **_solution_payload(solution))
    return out_path


def save_solution_mat(solution: SDVersionCSolution, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_path, _solution_payload(solution), do_compression=True)
    return out_path


def _build_primal_indices(problem: SDVersionCProblem) -> dict:
    return {
        "rho_idx": np.asarray([[1, problem.layout.n_rho]], dtype=np.int64),
        "alpha_idx": np.asarray([[problem.layout.alpha_index + 1, problem.layout.alpha_index + 1]], dtype=np.int64),
    }


def _build_dual_eq_indices(problem: SDVersionCProblem, dual_eq_size: int | None) -> dict | None:
    if dual_eq_size is None:
        return None
    n_force = int(problem.C_sparse.shape[0])
    if dual_eq_size == n_force:
        return {"force_balance": np.asarray([[1, n_force]], dtype=np.int64)}

    p_dim = int(problem.P_vm.shape[0])
    n_link_per_vertex = int(problem.layout.n_gauss * p_dim)
    expected = n_force + problem.n_vertices * n_link_per_vertex
    if dual_eq_size != expected:
        return {"force_balance": np.asarray([[1, dual_eq_size]], dtype=np.int64)}

    out = {"force_balance": np.asarray([[1, n_force]], dtype=np.int64)}
    cursor = n_force + 1
    for vertex_index in range(problem.n_vertices):
        end = cursor + n_link_per_vertex - 1
        out[f"y_link_vertex{vertex_index + 1}"] = np.asarray([[cursor, end]], dtype=np.int64)
        cursor = end + 1
    return out


def _build_dual_ineq_indices(problem: SDVersionCProblem) -> dict:
    out = {}
    cursor = 1
    for vertex_index in range(problem.n_vertices):
        end = cursor + problem.layout.n_gauss - 1
        out[f"yield_vertex{vertex_index + 1}"] = np.asarray([[cursor, end]], dtype=np.int64)
        cursor = end + 1
    return out


def _empty_solution(problem: SDVersionCProblem, status: str, solver_name: str, solve_time: float, objective_value):
    return SDVersionCSolution(
        status=status,
        solver_name=solver_name,
        solve_time=solve_time,
        objective_value=objective_value,
        alpha=None,
        primal_variables=None,
        primal_variable_index=None,
        dual_variables_eq=None,
        dual_variables_ineq=None,
        dual_variable_index_eq=None,
        dual_variable_index_ineq=None,
        residual_stress=None,
        yield_ratio=None,
        equilibrium_residual=None,
        active_gauss_count=None,
        active_gauss_fraction=None,
        problem_summary=problem.summary(),
        metadata={"cvxpy_status": str(status)},
    )


def postprocess_version_c(
    problem: SDVersionCProblem,
    rho,
    alpha: float,
    dual_variables_eq=None,
    dual_variables_ineq=None,
    active_tol: float = 1e-6,
    equilibrium_tol: float = 1e-8,
    status: str = "postprocessed",
    solver_name: str = "manual",
    solve_time: float = 0.0,
    objective_value: float | None = None,
    metadata: dict | None = None,
) -> SDVersionCSolution:
    """Post-process a candidate Version-C solution."""

    rho = np.asarray(rho, dtype=np.float64)
    if rho.size == 1:
        rho = np.full(
            (problem.layout.n_gauss, problem.layout.n_str),
            float(rho.reshape(-1)[0]),
            dtype=np.float64,
        )
    else:
        rho = rho.reshape(problem.layout.n_gauss, problem.layout.n_str)
    alpha = float(alpha)

    yield_ratio = problem.evaluate_yield_ratios(rho, alpha)
    equilibrium_residual = problem.C_sparse @ rho.reshape(-1)
    active_mask = np.any(yield_ratio >= (1.0 - active_tol), axis=1)
    active_gauss_count = int(np.count_nonzero(active_mask))
    active_gauss_fraction = float(active_gauss_count / problem.layout.n_gauss)

    info = {
        "equilibrium_linf": float(np.max(np.abs(equilibrium_residual))),
        "equilibrium_l2": float(np.linalg.norm(equilibrium_residual)),
        "equilibrium_ok": bool(np.max(np.abs(equilibrium_residual)) <= equilibrium_tol),
    }
    if metadata:
        info.update(metadata)

    primal_variables = np.concatenate([rho.reshape(-1), np.asarray([alpha], dtype=np.float64)])
    dual_eq = None if dual_variables_eq is None else np.asarray(dual_variables_eq, dtype=np.float64).reshape(-1)
    dual_ineq = None if dual_variables_ineq is None else np.asarray(dual_variables_ineq, dtype=np.float64).reshape(-1)

    return SDVersionCSolution(
        status=status,
        solver_name=solver_name,
        solve_time=float(solve_time),
        objective_value=float(alpha if objective_value is None else objective_value),
        alpha=alpha,
        primal_variables=primal_variables,
        primal_variable_index=_build_primal_indices(problem),
        dual_variables_eq=dual_eq,
        dual_variables_ineq=dual_ineq,
        dual_variable_index_eq=_build_dual_eq_indices(problem, None if dual_eq is None else dual_eq.size),
        dual_variable_index_ineq=None if dual_ineq is None else _build_dual_ineq_indices(problem),
        residual_stress=rho,
        yield_ratio=yield_ratio,
        equilibrium_residual=np.asarray(equilibrium_residual, dtype=np.float64),
        active_gauss_count=active_gauss_count,
        active_gauss_fraction=active_gauss_fraction,
        problem_summary=problem.summary(),
        metadata=info,
    )


def solve_version_c(
    problem: SDVersionCProblem,
    solver: str = "SCS",
    solver_options: dict | None = None,
    alpha_lb: float = 0.0,
) -> SDVersionCSolution:
    """Solve the Version-C shakedown problem with CVXPY."""

    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "solve_version_c requires the optional dependency 'cvxpy'. "
            "Install CVXPY in the runtime environment before calling this solver."
        ) from exc

    solver_options = {} if solver_options is None else dict(solver_options)
    n_gauss = problem.layout.n_gauss
    n_str = problem.layout.n_str

    rho = cp.Variable((n_gauss, n_str))
    alpha = cp.Variable(nonneg=True)
    rho_flat = cp.reshape(rho, (problem.layout.n_rho,), order="C")

    eq_constraint = problem.C_sparse @ rho_flat == problem.b_eq
    constraints = [eq_constraint, alpha >= float(alpha_lb)]

    p_vm_t = problem.P_vm.T
    yield_constraints = []
    for vertex_index in range(problem.n_vertices):
        sigma_shift = problem.sigma_vertices[:, :, vertex_index] @ p_vm_t
        total_dev = rho @ p_vm_t + alpha * sigma_shift
        c = cp.norm(total_dev, 2, axis=1) <= problem.ele_yield
        constraints.append(c)
        yield_constraints.append(c)

    objective = cp.Maximize(alpha)
    model = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    model.solve(solver=getattr(cp, solver), **solver_options)
    solve_time = time.perf_counter() - t0

    if rho.value is None or alpha.value is None:
        return _empty_solution(
            problem=problem,
            status=str(model.status),
            solver_name=solver,
            solve_time=solve_time,
            objective_value=None if model.value is None else float(model.value),
        )

    dual_eq = np.asarray(eq_constraint.dual_value, dtype=np.float64).reshape(-1)
    dual_ineq = np.concatenate(
        [np.asarray(c.dual_value, dtype=np.float64).reshape(-1) for c in yield_constraints],
        axis=0,
    )

    return postprocess_version_c(
        problem=problem,
        rho=np.asarray(rho.value, dtype=np.float64),
        alpha=float(alpha.value),
        dual_variables_eq=dual_eq,
        dual_variables_ineq=dual_ineq,
        status=str(model.status),
        solver_name=solver,
        solve_time=solve_time,
        objective_value=None if model.value is None else float(model.value),
        metadata={
            "cvxpy_status": str(model.status),
            "cvxpy_optval": None if model.value is None else float(model.value),
        },
    )
