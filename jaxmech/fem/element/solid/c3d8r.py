"""
jaxmech.fem.element.solid.c3d8r — 8-node trilinear hexahedron with reduced integration.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jaxmech.fem.element.solid.c3d8 import C3D8Kernel, _build_B_3d
from jaxmech.fem.element.shape_functions import hex8_shape_functions
from jaxmech.fem.quadrature.gauss import get_gauss_3d

# Reduced integration (1x1x1)
_QUAD_POINTS_R, _QUAD_WEIGHTS_R = get_gauss_3d(1)

# Precompute shape functions and derivatives at the single Gauss point
_N_R, _DN_DXI_R = jax.vmap(hex8_shape_functions)(_QUAD_POINTS_R)


class C3D8RKernel(C3D8Kernel):
    """8-node trilinear hexahedron element with reduced integration (C3D8R)."""

    ele_type = "C3D8R"
    n_gauss = 1

    def __init__(self, variant: str = "standard"):
        # B-bar not applicable for single GP
        super().__init__(variant="standard")

    def jacobian(self, node_coords: jnp.ndarray, gp_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dN_dxi_gp = _DN_DXI_R[gp_idx]
        J = jnp.dot(dN_dxi_gp, node_coords)
        detJ = jnp.linalg.det(J)
        J_inv = jnp.linalg.inv(J)
        return J, J_inv, detJ

    def sym_grad_voigt_E(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """Engineering Voigt B matrix at reduced integration point. (6, 24)"""
        dN_dxi_gp = _DN_DXI_R[gp_idx]
        _, J_inv, _ = self.jacobian(node_coords, gp_idx)
        dN_dx = jnp.dot(J_inv, dN_dxi_gp)
        return _build_B_3d(dN_dx, self.n_nodes)

    def sym_grad_voigt_E_all(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Return engineering-Voigt B matrices for all Gauss points."""
        J_all = jnp.einsum("gik,kj->gij", _DN_DXI_R, node_coords)
        J_inv_all = jnp.linalg.inv(J_all)
        dN_dx_all = jnp.einsum("gij,gjk->gik", J_inv_all, _DN_DXI_R)
        return jax.vmap(lambda dN_dx: _build_B_3d(dN_dx, self.n_nodes))(dN_dx_all)

    def gauss_geometry_all(self, node_coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return Gauss coordinates and integration volumes for one element."""
        gp_coords = jnp.einsum("gn,nc->gc", _N_R, node_coords)
        J_all = jnp.einsum("gik,kj->gij", _DN_DXI_R, node_coords)
        detJ_all = jnp.linalg.det(J_all)
        return gp_coords, detJ_all * _QUAD_WEIGHTS_R

    def gauss_info(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _QUAD_POINTS_R, _QUAD_WEIGHTS_R
