"""Common load-domain helpers shared by shakedown builders."""

from __future__ import annotations

import numpy as np


def default_phi_deg_for_equal_three_loads() -> float:
    return float(np.degrees(np.arcsin(1.0 / np.sqrt(3.0))))


def resolve_load_point(
    n_independent: int,
    load_scale,
    theta_deg: float | None,
    phi_deg: float | None,
) -> tuple[np.ndarray, dict]:
    """Resolve the loaded-state coefficients of 1 to 3 independent loads."""
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
        return np.asarray([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64), {
            "load_param_mode": "theta_deg",
            "theta_deg": theta,
            "phi_deg": np.nan,
        }

    if n_independent == 3:
        theta = 45.0 if theta_deg is None else float(theta_deg)
        phi = default_phi_deg_for_equal_three_loads() if phi_deg is None else float(phi_deg)
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        return np.asarray(
            [
                np.cos(phi_rad) * np.cos(theta_rad),
                np.cos(phi_rad) * np.sin(theta_rad),
                np.sin(phi_rad),
            ],
            dtype=np.float64,
        ), {
            "load_param_mode": "spherical_angles_deg",
            "theta_deg": theta,
            "phi_deg": phi,
        }

    raise ValueError(f"Unsupported n_independent={n_independent}.")


def allowed_vertex_counts(n_independent: int) -> tuple[int, ...]:
    mapping = {
        1: (1, 2),
        2: (1, 2, 4),
        3: (1, 2, 8),
    }
    if n_independent not in mapping:
        raise ValueError(f"Expected 1 to 3 independent loads, got {n_independent}.")
    return mapping[n_independent]


def binary_vertex_table(n_independent: int) -> np.ndarray:
    """Return binary hyper-rectangle vertices in lexicographic order."""
    bits = np.asarray(list(np.ndindex(*(2,) * n_independent)), dtype=np.float64)
    return bits.T


def build_vertex_load_matrix(
    n_independent: int,
    load_scale,
    n_vertices: int | None,
    theta_deg: float | None,
    phi_deg: float | None,
    R_ratios,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the load point and its max/min vertex matrix."""
    load_point, param_meta = resolve_load_point(
        n_independent=n_independent,
        load_scale=load_scale,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
    )
    allowed = allowed_vertex_counts(n_independent)
    if n_vertices is None:
        n_vertices = allowed[-1]
    n_vertices = int(n_vertices)
    if n_vertices not in allowed:
        allowed_str = ", ".join(str(v) for v in allowed)
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

    r_max = load_point.copy()
    r_min = r_full * load_point

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
        corners = binary_vertex_table(n_independent)
        vertex_load_matrix = r_min[:, None] + (r_max - r_min)[:, None] * corners

    param_meta["R_ratios"] = r_full
    return load_point, vertex_load_matrix, param_meta
