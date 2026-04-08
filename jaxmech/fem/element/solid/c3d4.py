"""
jaxmech.fem.element.solid.c3d4 — 4-node constant strain tetrahedron.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from jaxmech.fem.element.core import ElementKernel
from jaxmech.fem.element.solid.c3d8 import _build_B_3d


# Shape function derivatives are constant for linear tetrahedron
_DN_DXI_TET4 = jnp.array([
    [-1.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0],
], dtype=jnp.float64)

# 1-point rule: centroid (1/4, 1/4, 1/4), weight = 1/6
_QUAD_POINTS = jnp.array([[0.25, 0.25, 0.25]], dtype=jnp.float64)
_QUAD_WEIGHTS = jnp.array([1.0 / 6.0], dtype=jnp.float64)
_N_TET4 = jnp.array([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float64)


class C3D4Kernel(ElementKernel):
    """4-node constant strain tetrahedron element (C3D4)."""

    ele_type = "C3D4"
    cell_type = "tetrahedron"
    n_nodes = 4
    n_dim = 3
    n_gauss = 1
    n_str = 6

    def jacobian(self, node_coords: jnp.ndarray, gp_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute Jacobian mapping (constant for tetrahedron, gp_idx ignored)."""
        J = jnp.dot(_DN_DXI_TET4, node_coords)  # (3, 4) @ (4, 3) -> (3, 3)
        detJ = jnp.linalg.det(J)
        J_inv = jnp.linalg.inv(J)
        return J, J_inv, detJ

    def sym_grad_voigt_E(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """Engineering Voigt B matrix (constant, gp_idx ignored). (6, 12)"""
        _, J_inv, _ = self.jacobian(node_coords, gp_idx)
        dN_dx = jnp.dot(J_inv, _DN_DXI_TET4)
        return _build_B_3d(dN_dx, self.n_nodes)

    def gauss_geometry_all(self, node_coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return Gauss coordinates and integration volumes for one element."""
        gp_coords = jnp.einsum("gn,nc->gc", _N_TET4, node_coords)
        J = jnp.dot(_DN_DXI_TET4, node_coords)
        detJ = jnp.linalg.det(J)
        return gp_coords, detJ * _QUAD_WEIGHTS

    def gauss_info(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _QUAD_POINTS, _QUAD_WEIGHTS
