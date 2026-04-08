"""
jaxmech.fem.element.solid.c3d6 — 6-node triangular prism element.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jaxmech.fem.element.core import ElementKernel
from jaxmech.fem.element.solid.c3d8 import _build_B_3d
from jaxmech.fem.element.shape_functions import tri3_shape_functions
from jaxmech.fem.quadrature.gauss import get_gauss_1d
from jaxmech.fem.quadrature.triangle import get_triangle_quadrature

# 2-point rule for prism (1-point triangle x 2-point line)
_TRI_PTS, _TRI_WTS = get_triangle_quadrature(1)
_LINE_PTS, _LINE_WTS = get_gauss_1d(2)

# Build combined quadrature
_QUAD_POINTS = jnp.array([
    [t[0], t[1], l] for t in _TRI_PTS for l in _LINE_PTS
], dtype=jnp.float64)

_QUAD_WEIGHTS = jnp.array([
    tw * lw for tw in _TRI_WTS for lw in _LINE_WTS
], dtype=jnp.float64)


def prism6_shape_functions(xi_eta_zeta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shape functions for 6-node triangular prism.
    Domain is unit triangle in (xi, eta) and [-1,1] in zeta.

    Nodes:
    z=-1: (0,0,-1), (1,0,-1), (0,1,-1)
    z= 1: (0,0, 1), (1,0, 1), (0,1, 1)
    """
    xi, eta, zeta = xi_eta_zeta[0], xi_eta_zeta[1], xi_eta_zeta[2]

    # Triangle part
    Nt = jnp.array([1.0 - xi - eta, xi, eta], dtype=jnp.float64)
    dNt_dxi = jnp.array([
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ], dtype=jnp.float64)

    # Line part
    Nz = jnp.array([0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)], dtype=jnp.float64)
    dNz_dzeta = jnp.array([-0.5, 0.5], dtype=jnp.float64)

    # Combined N
    N = jnp.array([
        Nt[0]*Nz[0], Nt[1]*Nz[0], Nt[2]*Nz[0],
        Nt[0]*Nz[1], Nt[1]*Nz[1], Nt[2]*Nz[1]
    ], dtype=jnp.float64)

    # Combined dN_dxi (3, 6)
    dN_dxi = jnp.zeros((3, 6), dtype=jnp.float64)

    # d/dxi
    dN_dxi = dN_dxi.at[0, :3].set(dNt_dxi[0, :] * Nz[0])
    dN_dxi = dN_dxi.at[0, 3:].set(dNt_dxi[0, :] * Nz[1])

    # d/deta
    dN_dxi = dN_dxi.at[1, :3].set(dNt_dxi[1, :] * Nz[0])
    dN_dxi = dN_dxi.at[1, 3:].set(dNt_dxi[1, :] * Nz[1])

    # d/dzeta
    dN_dxi = dN_dxi.at[2, :3].set(Nt * dNz_dzeta[0])
    dN_dxi = dN_dxi.at[2, 3:].set(Nt * dNz_dzeta[1])

    return N, dN_dxi

_N, _DN_DXI = jax.vmap(prism6_shape_functions)(_QUAD_POINTS)


class C3D6Kernel(ElementKernel):
    """6-node triangular prism element (C3D6)."""

    ele_type = "C3D6"
    cell_type = "wedge"
    n_nodes = 6
    n_dim = 3
    n_gauss = 2
    n_str = 6

    def jacobian(self, node_coords: jnp.ndarray, gp_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dN_dxi_gp = _DN_DXI[gp_idx]
        J = jnp.dot(dN_dxi_gp, node_coords)  # (3, 6) @ (6, 3) -> (3, 3)
        detJ = jnp.linalg.det(J)
        J_inv = jnp.linalg.inv(J)
        return J, J_inv, detJ

    def sym_grad_voigt_E(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """Engineering Voigt B matrix at a Gauss point. (6, 18)"""
        dN_dxi_gp = _DN_DXI[gp_idx]
        _, J_inv, _ = self.jacobian(node_coords, gp_idx)
        dN_dx = jnp.dot(J_inv, dN_dxi_gp)
        return _build_B_3d(dN_dx, self.n_nodes)

    def gauss_geometry_all(self, node_coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return Gauss coordinates and integration volumes for one element."""
        gp_coords = jnp.einsum("gn,nc->gc", _N, node_coords)
        J_all = jnp.einsum("gik,kj->gij", _DN_DXI, node_coords)
        detJ_all = jnp.linalg.det(J_all)
        return gp_coords, detJ_all * _QUAD_WEIGHTS

    def gauss_info(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _QUAD_POINTS, _QUAD_WEIGHTS
