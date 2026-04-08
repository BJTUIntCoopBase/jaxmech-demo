"""
jaxmech.fem.quadrature.triangle — Quadrature rules for triangles.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def get_triangle_quadrature(order: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Quadrature rules for a unit reference triangle.
    Vertices expected at (0,0), (1,0), (0,1).

    Parameters
    ----------
    order : int
        Polynomial integration order (supports 1, 2, 3 corresponding to 1, 3, 4 points).

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        points: shape (n_points, 2), coordinates (xi, eta)
        weights: shape (n_points,)
    """
    # 1-point rule (integrates linear exact)
    if order == 1:
        points = jnp.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=jnp.float64)
        weights = jnp.array([0.5], dtype=jnp.float64)

    # 3-point rule (integrates quadratic exact)
    elif order == 2:
        points = jnp.array([
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0]
        ], dtype=jnp.float64)
        weights = jnp.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=jnp.float64)

    # 4-point rule (symmetric)
    elif order == 3:
        points = jnp.array([
            [1.0 / 3.0, 1.0 / 3.0],
            [0.2, 0.2],
            [0.6, 0.2],
            [0.2, 0.6]
        ], dtype=jnp.float64)
        # Note: weights sum to 0.5 (area of unit triangle)
        weights = jnp.array([
            -27.0 / 96.0,
            25.0 / 96.0,
            25.0 / 96.0,
            25.0 / 96.0
        ], dtype=jnp.float64)

    else:
        raise ValueError(f"Triangle quadrature order {order} not implemented.")

    return points, weights
