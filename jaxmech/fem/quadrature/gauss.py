"""
jaxmech.fem.quadrature.gauss — Standard Gauss-Legendre quadrature rules.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def get_gauss_1d(order: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """1D Gauss-Legendre quadrature on [-1, 1].

    Parameters
    ----------
    order : int
        Number of integration points (1 to 3).

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        points: shape (order,)
        weights: shape (order,)
    """
    if order == 1:
        points = jnp.array([0.0], dtype=jnp.float64)
        weights = jnp.array([2.0], dtype=jnp.float64)
    elif order == 2:
        val = jnp.sqrt(1.0 / 3.0)
        points = jnp.array([-val, val], dtype=jnp.float64)
        weights = jnp.array([1.0, 1.0], dtype=jnp.float64)
    elif order == 3:
        val = jnp.sqrt(3.0 / 5.0)
        points = jnp.array([-val, 0.0, val], dtype=jnp.float64)
        weights = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=jnp.float64)
    else:
        raise ValueError(f"1D Gauss order {order} not implemented.")
    return points, weights


def get_gauss_2d(order: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """2D Gauss-Legendre quadrature on [-1, 1] x [-1, 1].

    Parameters
    ----------
    order : int
        Number of points per direction, total points = order^2.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        points: shape (order^2, 2)
        weights: shape (order^2,)
    """
    p1d, w1d = get_gauss_1d(order)
    points = jnp.array([[x, y] for x in p1d for y in p1d], dtype=jnp.float64)
    weights = jnp.array([wx * wy for wx in w1d for wy in w1d], dtype=jnp.float64)
    return points, weights


def get_gauss_3d(order: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """3D Gauss-Legendre quadrature on [-1, 1]^3.

    Parameters
    ----------
    order : int
        Number of points per direction, total points = order^3.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        points: shape (order^3, 3)
        weights: shape (order^3,)
    """
    p1d, w1d = get_gauss_1d(order)
    points = jnp.array([[x, y, z] for x in p1d for y in p1d for z in p1d], dtype=jnp.float64)
    weights = jnp.array([wx * wy * wz for wx in w1d for wy in w1d for wz in w1d], dtype=jnp.float64)
    return points, weights
