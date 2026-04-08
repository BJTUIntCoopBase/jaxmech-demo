"""
SD_VersionC
===========
Construct the lower-bound shakedown problem for the direct C-matrix
formulation.

Current scope:
- Version-C only
- 1 to 3 independent elastic load cases
- user-controlled vertex count
- MATLAB-aligned vertex generation for multi-vertex cases
- problem construction only, no optimizer call yet

The optimization variables are

    x = [rho_flat, alpha]

where ``rho_flat`` stacks the residual stress at all Gauss points in the
project Voigt ordering. The convex lower-bound problem is

    maximize    alpha
    subject to  C_sparse @ rho_flat = 0
                ||P_vm @ (alpha * sigma_v(r) + rho_r)||^2 <= sigma_y(r)^2
                for every Gauss point r and every load vertex v
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack

from jaxmech.optimize.cvxpy.var_convert import VarConverter


@dataclass(frozen=True)
class SDVersionCVariableLayout:
    """Variable indexing for ``x = [rho_flat, alpha]``."""

    n_gauss: int
    n_str: int

    @property
    def n_rho(self) -> int:
        return int(self.n_gauss * self.n_str)

    @property
    def alpha_index(self) -> int:
        return int(self.n_rho)

    @property
    def total_size(self) -> int:
        return int(self.n_rho + 1)

    def gauss_slice(self, gauss_id: int) -> slice:
        if gauss_id < 0 or gauss_id >= self.n_gauss:
            raise IndexError(f"gauss_id={gauss_id} is out of range [0, {self.n_gauss}).")
        start = gauss_id * self.n_str
        return slice(start, start + self.n_str)


@dataclass(frozen=True)
class SDVersionCProblem:
    """Solver-agnostic Version-C lower-bound problem data."""

    formulation: str
    ele_type: str
    n_independent_loads: int
    n_vertices: int
    layout: SDVersionCVariableLayout
    C_sparse: csr_matrix
    A_eq: csr_matrix
    b_eq: np.ndarray
    objective: np.ndarray
    objective_sense: str
    sigma_vertices: np.ndarray
    w_vertices: np.ndarray
    ele_yield: np.ndarray
    P_vm: np.ndarray
    gauss_coords: np.ndarray
    gauss_vols: np.ndarray
    free_dofs: np.ndarray
    independent_case_ids: np.ndarray
    load_scale: np.ndarray
    vertex_load_matrix: np.ndarray
    metadata: dict

    def rho_slice(self, gauss_id: int) -> slice:
        return self.layout.gauss_slice(gauss_id)

    def split_x(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Split a full decision vector into ``rho`` and ``alpha``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.layout.total_size:
            raise ValueError(
                f"Expected decision vector of length {self.layout.total_size}, got {x.size}."
            )
        rho = x[: self.layout.n_rho].reshape(self.layout.n_gauss, self.layout.n_str)
        alpha = float(x[self.layout.alpha_index])
        return rho, alpha

    def compute_total_stress(self, rho, alpha: float, vertex_index: int) -> np.ndarray:
        """Return total stress ``alpha * sigma_vertex + rho`` at all Gauss points."""
        if vertex_index < 0 or vertex_index >= self.n_vertices:
            raise IndexError(f"vertex_index={vertex_index} is out of range [0, {self.n_vertices}).")
        rho = np.asarray(rho, dtype=np.float64).reshape(self.layout.n_gauss, self.layout.n_str)
        return alpha * self.sigma_vertices[:, :, vertex_index] + rho

    def evaluate_yield_ratios(self, rho, alpha: float) -> np.ndarray:
        """Return ``sigma_eq / sigma_y`` for every Gauss point and vertex."""
        rho = np.asarray(rho, dtype=np.float64).reshape(self.layout.n_gauss, self.layout.n_str)
        ratios = np.empty((self.layout.n_gauss, self.n_vertices), dtype=np.float64)
        for vertex_index in range(self.n_vertices):
            sigma_total = self.compute_total_stress(rho, alpha, vertex_index)
            equiv_sq = np.sum((sigma_total @ self.P_vm.T) ** 2, axis=1)
            ratios[:, vertex_index] = np.sqrt(np.maximum(equiv_sq, 0.0)) / self.ele_yield
        return ratios

    def summary(self) -> dict:
        """Return a compact shape summary for debugging."""
        return {
            "formulation": self.formulation,
            "ele_type": self.ele_type,
            "n_independent_loads": self.n_independent_loads,
            "n_vertices": self.n_vertices,
            "n_gauss": self.layout.n_gauss,
            "n_str": self.layout.n_str,
            "n_free": int(self.C_sparse.shape[0]),
            "rho_size": self.layout.n_rho,
            "total_var_size": self.layout.total_size,
            "A_eq_shape": tuple(self.A_eq.shape),
            "C_sparse_shape": tuple(self.C_sparse.shape),
            "sigma_vertices_shape": tuple(self.sigma_vertices.shape),
            "vertex_load_matrix_shape": tuple(self.vertex_load_matrix.shape),
            "P_vm_shape": tuple(self.P_vm.shape),
        }


def _load_fem_input(fem_input) -> dict:
    if isinstance(fem_input, dict):
        return fem_input
    raise TypeError(
        "build_lower_bound_problem expects a normalized fem_input dict. "
        "MAT-first loading belongs to jaxmech.modules.shakedown.preprocess."
    )


def _infer_ele_type(n_str: int) -> str:
    mapping = {
        6: "C3D8",
        4: "CPE4",
        3: "CPS4",
    }
    if n_str not in mapping:
        raise ValueError(f"Cannot infer element type from n_str={n_str}.")
    return mapping[n_str]


def _available_sigma_cases(fem_data: dict) -> np.ndarray:
    base = np.asarray(fem_data["sigma_E"], dtype=np.float64)
    if base.ndim != 2:
        raise ValueError(f"sigma_E must have shape (n_gauss, n_str), got {base.shape}.")

    all_cases = base[:, :, None]
    if "sigma_E_cases" not in fem_data:
        return all_cases

    extra = np.asarray(fem_data["sigma_E_cases"], dtype=np.float64)
    if extra.ndim != 3 or extra.shape[:2] != base.shape:
        raise ValueError(
            "sigma_E_cases must have shape (n_gauss, n_str, n_cases). "
            f"Got {extra.shape}, expected {base.shape + ('n_cases',)}."
        )
    return np.concatenate([all_cases, extra], axis=2)


def _default_phi_deg_for_equal_three_loads() -> float:
    return float(np.degrees(np.arcsin(1.0 / np.sqrt(3.0))))


def _resolve_load_point(
    n_independent: int,
    load_scale,
    theta_deg: float | None,
    phi_deg: float | None,
) -> tuple[np.ndarray, dict]:
    """Resolve the loaded-state coefficients ``[a1, a2, a3]``.

    Rules:
    - 1 load  -> fixed to [1]
    - 2 loads -> explicit [a1, a2] or [cos(theta), sin(theta)]
    - 3 loads -> explicit [a1, a2, a3] or spherical coordinates
    """
    if n_independent == 1:
        if load_scale is not None:
            load_scale = np.asarray(load_scale, dtype=np.float64).reshape(-1)
            if load_scale.size not in (0, 1):
                raise ValueError("For one load, load_scale must be omitted or length 1.")
        if theta_deg is not None or phi_deg is not None:
            raise ValueError("theta_deg/phi_deg are not used for one independent load.")
        return np.asarray([1.0], dtype=np.float64), {
            "load_param_mode": "single_load_fixed",
            "theta_deg": np.nan,
            "phi_deg": np.nan,
        }

    if load_scale is not None:
        resolved = np.asarray(load_scale, dtype=np.float64).reshape(-1)
        if resolved.size != n_independent:
            raise ValueError(
                f"load_scale must have length {n_independent}, got {resolved.size}."
            )
        return resolved, {
            "load_param_mode": "explicit",
            "theta_deg": np.nan if theta_deg is None else float(theta_deg),
            "phi_deg": np.nan if phi_deg is None else float(phi_deg),
        }

    if n_independent == 2:
        theta = 45.0 if theta_deg is None else float(theta_deg)
        if phi_deg is not None:
            raise ValueError("phi_deg is not used for two independent loads.")
        theta_rad = np.deg2rad(theta)
        resolved = np.asarray([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64)
        return resolved, {
            "load_param_mode": "theta_deg",
            "theta_deg": theta,
            "phi_deg": np.nan,
        }

    if n_independent == 3:
        theta = 45.0 if theta_deg is None else float(theta_deg)
        phi = _default_phi_deg_for_equal_three_loads() if phi_deg is None else float(phi_deg)
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        resolved = np.asarray(
            [
                np.cos(phi_rad) * np.cos(theta_rad),
                np.cos(phi_rad) * np.sin(theta_rad),
                np.sin(phi_rad),
            ],
            dtype=np.float64,
        )
        return resolved, {
            "load_param_mode": "spherical_angles_deg",
            "theta_deg": theta,
            "phi_deg": phi,
        }

    raise ValueError(f"Unsupported n_independent={n_independent}.")


def _allowed_vertex_counts(n_independent: int) -> tuple[int, ...]:
    mapping = {
        1: (1, 2),
        2: (1, 2, 4),
        3: (1, 2, 8),
    }
    if n_independent not in mapping:
        raise ValueError(f"Expected 1 to 3 independent loads, got {n_independent}.")
    return mapping[n_independent]


def _binary_vertex_table(n_independent: int) -> np.ndarray:
    """Return binary vertex indicators in lexicographic order.

    Examples
    --------
    2 loads -> columns: (0,0), (0,1), (1,0), (1,1)
    3 loads -> columns: (0,0,0), ..., (1,1,1)
    """
    bits = np.asarray(list(np.ndindex(*(2,) * n_independent)), dtype=np.float64)
    return bits.T


def _available_case_ids(all_cases: np.ndarray, independent_case_ids) -> np.ndarray:
    if independent_case_ids is None:
        return np.asarray([0], dtype=np.int32)

    case_ids = np.asarray(independent_case_ids, dtype=np.int32).reshape(-1)
    if case_ids.size < 1 or case_ids.size > 3:
        raise ValueError(
            f"Expected 1 to 3 independent_case_ids, got {case_ids.size}."
        )
    if np.any(case_ids < 0) or np.any(case_ids >= all_cases.shape[2]):
        raise IndexError(
            f"independent_case_ids={case_ids.tolist()} exceed available stress cases "
            f"[0, {all_cases.shape[2]})."
        )
    return case_ids


def _normalize_independent_stresses(
    fem_data: dict,
    independent_stresses,
    independent_case_ids,
) -> tuple[np.ndarray, np.ndarray]:
    n_gauss = int(fem_data["n_gauss"])
    n_str = int(fem_data["n_str"])

    if independent_stresses is not None:
        independent_stresses = np.asarray(independent_stresses, dtype=np.float64)
        if independent_stresses.ndim != 3:
            raise ValueError(
                "independent_stresses must be a 3D array with shape "
                "(n_gauss, n_str, n_independent) or (n_independent, n_gauss, n_str)."
            )
        if independent_stresses.shape[:2] == (n_gauss, n_str):
            cases = independent_stresses
        elif independent_stresses.shape[1:] == (n_gauss, n_str):
            cases = np.transpose(independent_stresses, (1, 2, 0))
        else:
            cases = None
        if cases is None:
            raise ValueError(
                "independent_stresses must have shape (n_gauss, n_str, n_independent) "
                f"or (n_independent, n_gauss, n_str). Got {independent_stresses.shape}."
            )
        if cases.shape[2] < 1 or cases.shape[2] > 3:
            raise ValueError(
                f"Expected 1 to 3 independent stresses, got {cases.shape[2]}."
            )
        case_ids = -np.arange(1, cases.shape[2] + 1, dtype=np.int32)
        return cases, case_ids

    all_cases = _available_sigma_cases(fem_data)
    case_ids = _available_case_ids(all_cases, independent_case_ids)
    return all_cases[:, :, case_ids], case_ids


def _build_sigma_vertices(
    independent_stresses: np.ndarray,
    load_scale,
    n_vertices: int | None,
    theta_deg: float | None,
    phi_deg: float | None,
    R_ratios,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n_independent = independent_stresses.shape[2]
    if n_independent < 1 or n_independent > 3:
        raise ValueError(
            f"Expected 1 to 3 independent stresses, got {n_independent}."
        )

    load_point, param_meta = _resolve_load_point(
        n_independent=n_independent,
        load_scale=load_scale,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )
    allowed_vertex_counts = _allowed_vertex_counts(n_independent)
    if n_vertices is None:
        n_vertices = allowed_vertex_counts[-1]
    n_vertices = int(n_vertices)
    if n_vertices not in allowed_vertex_counts:
        allowed_str = ", ".join(str(v) for v in allowed_vertex_counts)
        raise ValueError(
            f"For {n_independent} independent loads, n_vertices must be one of "
            f"{{{allowed_str}}}; got {n_vertices}."
        )

    r_vec = np.asarray(R_ratios, dtype=np.float64).reshape(-1)
    if r_vec.size < 1 or r_vec.size > 3:
        raise ValueError(f"R_ratios must contain 1 to 3 values, got {r_vec.size}.")
    # Pad to n_independent with zeros if shorter, truncate if longer
    r_full = np.zeros(n_independent, dtype=np.float64)
    take = min(r_vec.size, n_independent)
    r_full[:take] = r_vec[:take]
    r_min = r_full * load_point
    r_max = load_point.copy()

    if n_vertices == 1:
        vertex_load_matrix = r_max.reshape(n_independent, 1)
    elif n_vertices == 2:
        vertex_load_matrix = np.column_stack([r_max, r_min])
    else:
        if n_vertices != 2 ** n_independent:
            raise ValueError(
                f"n_vertices={n_vertices} is not supported for {n_independent} loads. "
                f"Expected {2 ** n_independent} for binary corner generation."
            )
        corners = _binary_vertex_table(n_independent)
        vertex_load_matrix = r_min[:, None] + (r_max - r_min)[:, None] * corners

    sigma_vertices = np.einsum("gsi,iv->gsv", independent_stresses, vertex_load_matrix)
    param_meta["R_ratios"] = r_full
    return sigma_vertices, load_point, vertex_load_matrix, param_meta


def build_lower_bound_problem(
    fem_input,
    independent_stresses=None,
    independent_case_ids=None,
    load_scale=None,
    n_vertices: int | None = None,
    theta_deg: float | None = None,
    phi_deg: float | None = None,
    R_ratios=(0.0, 0.0, 0.0),
    ele_type: str | None = None,
) -> SDVersionCProblem:
    """Construct the Version-C lower-bound problem.

    Parameters
    ----------
    fem_input:
        Either a FEM bundle dict or a path to ``save_fem_data_npz`` output.
    independent_stresses:
        Optional explicit independent elastic stresses with shape
        ``(n_gauss, n_str, n_independent)`` or ``(n_independent, n_gauss, n_str)``.
        The allowed range is ``1 <= n_independent <= 3``.
    independent_case_ids:
        Optional indices into the available elastic case stack
        ``[sigma_E, *sigma_E_cases]``. If omitted, the default is ``[0]``,
        which reproduces the MATLAB 2-vertex case ``[sigma_E, 0]``.
    load_scale:
        Optional explicit loaded-state coefficients.
        - 1 load: ignored; the coefficient is fixed to ``1``.
        - 2 loads: optional explicit ``[a1, a2]``.
        - 3 loads: optional explicit ``[a1, a2, a3]``.
    n_vertices:
        Requested number of stress vertices. Allowed combinations:
        - 1 independent load : ``1`` or ``2``
        - 2 independent loads: ``1``, ``2``, or ``4``
        - 3 independent loads: ``1``, ``2``, or ``8``
    theta_deg:
        Optional angle in degrees used when ``load_scale`` is omitted.
        - 2 loads: ``[a1, a2] = [cos(theta), sin(theta)]``
        - 3 loads: spherical-coordinate azimuth angle
    phi_deg:
        Optional elevation angle in degrees for the 3-load spherical-coordinate
        parameterisation. Ignored for 1 or 2 loads.
    R_ratios:
        Length-3 vector of unloading ratios. ``R_i`` means
        ``sigma_i(min) / sigma_i(max)`` for load case ``i``. The provided
        elastic stresses correspond to the ``max`` side.
    ele_type:
        Optional element type override. Default is inferred from ``n_str``.
    """
    fem_data = _load_fem_input(fem_input)
    n_gauss = int(fem_data["n_gauss"])
    n_str = int(fem_data["n_str"])

    if ele_type is None:
        ele_type = _infer_ele_type(n_str)
    converter = VarConverter(ele_type)
    if converter.n_str != n_str:
        raise ValueError(
            f"ele_type={ele_type} expects n_str={converter.n_str}, but FEM bundle has n_str={n_str}."
        )

    independent_stresses, independent_case_ids = _normalize_independent_stresses(
        fem_data=fem_data,
        independent_stresses=independent_stresses,
        independent_case_ids=independent_case_ids,
    )
    sigma_vertices, load_scale, vertex_load_matrix, param_meta = _build_sigma_vertices(
        independent_stresses=independent_stresses,
        load_scale=load_scale,
        n_vertices=n_vertices,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        R_ratios=R_ratios,
    )

    C_sparse = fem_data["C_sparse"].tocsr()
    if C_sparse.shape[1] != n_gauss * n_str:
        raise ValueError(
            f"C_sparse shape {C_sparse.shape} is inconsistent with n_gauss={n_gauss}, n_str={n_str}."
        )

    ele_yield = np.asarray(fem_data["ele_yield"], dtype=np.float64).reshape(-1)
    if ele_yield.size != n_gauss:
        raise ValueError(f"ele_yield must have length {n_gauss}, got {ele_yield.size}.")
    if np.any(ele_yield <= 0.0):
        raise ValueError("ele_yield must be strictly positive at every Gauss point.")

    actual_n_vertices = int(sigma_vertices.shape[2])
    layout = SDVersionCVariableLayout(n_gauss=n_gauss, n_str=n_str)
    A_eq = hstack([C_sparse, csr_matrix((C_sparse.shape[0], 1), dtype=np.float64)], format="csr")
    b_eq = np.zeros((C_sparse.shape[0],), dtype=np.float64)
    objective = np.zeros((layout.total_size,), dtype=np.float64)
    objective[layout.alpha_index] = 1.0
    P_vm = converter.get_J2_matrix()
    w_vertices = np.column_stack([
        C_sparse @ sigma_vertices[:, :, vertex_id].reshape(-1)
        for vertex_id in range(sigma_vertices.shape[2])
    ])

    metadata = dict(fem_data.get("metadata", {}))
    metadata["available_case_count"] = int(_available_sigma_cases(fem_data).shape[2])
    metadata["n_independent_loads"] = int(independent_stresses.shape[2])
    metadata["n_vertices"] = actual_n_vertices
    metadata.update(param_meta)

    return SDVersionCProblem(
        formulation="lower_bound_version_c",
        ele_type=ele_type,
        n_independent_loads=int(independent_stresses.shape[2]),
        n_vertices=actual_n_vertices,
        layout=layout,
        C_sparse=C_sparse,
        A_eq=A_eq,
        b_eq=b_eq,
        objective=objective,
        objective_sense="max",
        sigma_vertices=np.asarray(sigma_vertices, dtype=np.float64),
        w_vertices=np.asarray(w_vertices, dtype=np.float64),
        ele_yield=ele_yield,
        P_vm=np.asarray(P_vm, dtype=np.float64),
        gauss_coords=np.asarray(fem_data["gauss_coords"], dtype=np.float64),
        gauss_vols=np.asarray(fem_data["gauss_vols"], dtype=np.float64).reshape(-1),
        free_dofs=np.asarray(fem_data["free_dofs"], dtype=np.int64).reshape(-1),
        independent_case_ids=np.asarray(independent_case_ids, dtype=np.int32),
        load_scale=np.asarray(load_scale, dtype=np.float64),
        vertex_load_matrix=np.asarray(vertex_load_matrix, dtype=np.float64),
        metadata=metadata,
    )
