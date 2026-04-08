"""
jaxmech.fem.element.solid.c3d8 — 8-node trilinear hexahedron element.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from jaxmech.fem.element.core import ElementKernel
from jaxmech.fem.element.shape_functions import hex8_shape_functions
from jaxmech.fem.quadrature.gauss import get_gauss_3d

# Standard full integration (2x2x2)
_QUAD_POINTS, _QUAD_WEIGHTS = get_gauss_3d(2)

# Precompute shape functions and derivatives at all Gauss points
_N, _DN_DXI = jax.vmap(hex8_shape_functions)(_QUAD_POINTS)


def _build_B_3d(dN_dx: jnp.ndarray, n_nodes: int) -> jnp.ndarray:
    """Assemble 3D B matrix from physical shape function derivatives.

    Voigt order: [xx, yy, zz, xy, yz, xz] (engineering convention, γ = 2ε).

    Parameters
    ----------
    dN_dx : (3, n_nodes)
    n_nodes : int

    Returns
    -------
    B : (6, n_nodes * 3)
    """
    n_dof = n_nodes * 3
    B = jnp.zeros((6, n_dof), dtype=jnp.float64)
    # xx
    B = B.at[0, 0::3].set(dN_dx[0, :])
    # yy
    B = B.at[1, 1::3].set(dN_dx[1, :])
    # zz
    B = B.at[2, 2::3].set(dN_dx[2, :])
    # xy
    B = B.at[3, 0::3].set(dN_dx[1, :])
    B = B.at[3, 1::3].set(dN_dx[0, :])
    # yz
    B = B.at[4, 1::3].set(dN_dx[2, :])
    B = B.at[4, 2::3].set(dN_dx[1, :])
    # xz
    B = B.at[5, 0::3].set(dN_dx[2, :])
    B = B.at[5, 2::3].set(dN_dx[0, :])
    return B


def _build_B_3d_optimized(dN_dx: jnp.ndarray, n_nodes: int) -> jnp.ndarray:
    """🚀 Optimized 3D B matrix assembly using direct array construction.
    
    Reduces memory allocations and intermediate operations.
    
    Parameters
    ----------
    dN_dx : jnp.ndarray
        Physical shape function derivatives, shape (3, n_nodes)
    n_nodes : int
        Number of nodes
    """
    # 🔧 修复：确保dN_dx形状正确
    dN_dx = jnp.asarray(dN_dx)
    if dN_dx.ndim == 1:
        # 如果是1维，假设是flatten的形式，reshape为(3, n_nodes)
        dN_dx = dN_dx.reshape(3, n_nodes)
    elif dN_dx.shape != (3, n_nodes):
        raise ValueError(f"Expected dN_dx shape (3, {n_nodes}), got {dN_dx.shape}")
    
    n_dof = n_nodes * 3
    
    # 🚀 优化4: 直接构造而非逐步修改
    # 安全地提取各个方向的导数
    dN_x, dN_y, dN_z = dN_dx[0, :], dN_dx[1, :], dN_dx[2, :]
    
    # 构造B矩阵的每一行
    row0 = jnp.zeros(n_dof).at[0::3].set(dN_x)  # xx
    row1 = jnp.zeros(n_dof).at[1::3].set(dN_y)  # yy  
    row2 = jnp.zeros(n_dof).at[2::3].set(dN_z)  # zz
    
    row3 = jnp.zeros(n_dof).at[0::3].set(dN_y).at[1::3].set(dN_x)  # xy
    row4 = jnp.zeros(n_dof).at[1::3].set(dN_z).at[2::3].set(dN_y)  # yz
    row5 = jnp.zeros(n_dof).at[0::3].set(dN_z).at[2::3].set(dN_x)  # xz
    
    return jnp.stack([row0, row1, row2, row3, row4, row5], axis=0)


def _build_B_3d_batch(dN_dx_all: jnp.ndarray, n_nodes: int) -> jnp.ndarray:
    """Build 3D B matrices for all Gauss points of one element.

    Parameters
    ----------
    dN_dx_all : (n_gp, 3, n_nodes)
        Physical shape function derivatives at all Gauss points.
    n_nodes : int

    Returns
    -------
    jnp.ndarray
        Batched B matrices with shape (n_gp, 6, n_nodes * 3).
    """
    return jax.vmap(lambda dN_dx: _build_B_3d_optimized(dN_dx, n_nodes))(dN_dx_all)


class C3D8Kernel(ElementKernel):
    """8-node trilinear hexahedron element (C3D8).

    Parameters
    ----------
    variant : str
        ``"standard"`` — full integration (2×2×2).
        ``"bbar"``    — B-bar method (volume locking mitigation).
    """

    ele_type = "C3D8"
    cell_type = "hexahedron"
    n_nodes = 8
    n_dim = 3
    n_gauss = 8
    n_str = 6

    def __init__(self, variant: str = "bbar"):
        self.variant = variant

    # ------------------------------------------------------------------
    # Core geometric methods
    # ------------------------------------------------------------------

    def jacobian(self, node_coords: jnp.ndarray, gp_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dN_dxi_gp = _DN_DXI[gp_idx]
        J = jnp.dot(dN_dxi_gp, node_coords)  # (3, 8) @ (8, 3) -> (3, 3)
        detJ = jnp.linalg.det(J)
        J_inv = jnp.linalg.inv(J)
        return J, J_inv, detJ

    def _standard_B(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """Standard B matrix at a Gauss point. (6, 24)"""
        dN_dxi_gp = _DN_DXI[gp_idx]
        _, J_inv, _ = self.jacobian(node_coords, gp_idx)
        dN_dx = jnp.dot(J_inv, dN_dxi_gp)
        return _build_B_3d(dN_dx, self.n_nodes)
    
    def _standard_B_optimized(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """🚀 Optimized standard B matrix at a Gauss point. (6, 24)"""
        dN_dxi_gp = _DN_DXI[gp_idx]
        _, J_inv, _ = self.jacobian(node_coords, gp_idx)
        dN_dx = jnp.dot(J_inv, dN_dxi_gp)
        try:
            return _build_B_3d_optimized(dN_dx, self.n_nodes)
        except Exception:
            # 回退到原始方法
            return _build_B_3d(dN_dx, self.n_nodes)

    def _bbar_B(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """B-bar matrix at a Gauss point (volume averaging for locking). (6, 24)"""
        B = self._standard_B(node_coords, gp_idx)
        # Deviatoric part of current GP
        trace_B = B[0:3, :]
        B_vol = jnp.zeros_like(B)
        B_vol = B_vol.at[0:3, :].set(jnp.sum(trace_B, axis=0, keepdims=True) / 3.0)
        B_dev = B - B_vol

        # Mean volumetric B over all GPs
        def compute_gp_vol(i, val):
            B_i = self._standard_B(node_coords, i)
            _, _, detJ_i = self.jacobian(node_coords, i)
            w_i = _QUAD_WEIGHTS[i]
            dV = detJ_i * w_i
            trace_B_i = jnp.sum(B_i[0:3, :], axis=0, keepdims=True) / 3.0
            return val[0] + dV, val[1] + trace_B_i * dV

        total_vol, vol_sum = jax.lax.fori_loop(
            0, self.n_gauss, compute_gp_vol,
            (0.0, jnp.zeros((1, 24), dtype=jnp.float64))
        )
        B_vol_mean = jnp.zeros_like(B)
        B_vol_mean = B_vol_mean.at[0:3, :].set(vol_sum / total_vol)

        return B_dev + B_vol_mean

    def sym_grad_voigt_E(self, node_coords: jnp.ndarray, gp_idx: int) -> jnp.ndarray:
        """Engineering Voigt B matrix, dispatching based on variant."""
        if self.variant == "bbar":
            return self._bbar_B(node_coords, gp_idx)
        return self._standard_B(node_coords, gp_idx)
    
    def _compute_all_bbar_matrices_optimized(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """🚀 Optimized B-bar computation for all Gauss points at once.
        
        This eliminates the O(n²) complexity of computing B-bar matrices
        by computing the volume average once and reusing it.
        
        Returns
        -------
        B_matrices : jnp.ndarray of shape (n_gauss, n_str, n_dof)
        """
        # 计算所有高斯点的标准B矩阵
        def compute_standard_B_and_vol(gp_idx):
            B_std = self._standard_B(node_coords, gp_idx)
            _, _, detJ = self.jacobian(node_coords, gp_idx)
            w = _QUAD_WEIGHTS[gp_idx]
            dV = detJ * w
            trace_B = jnp.sum(B_std[0:3, :], axis=0, keepdims=True) / 3.0
            return B_std, trace_B, dV
        
        # 批量计算所有高斯点
        B_std_all, trace_B_all, dV_all = jax.vmap(compute_standard_B_and_vol)(
            jnp.arange(self.n_gauss)
        )
        
        # 计算体积加权平均（只需要一次）
        total_vol = jnp.sum(dV_all)
        # 🔧 修复：确保广播维度正确
        # trace_B_all 形状: (n_gauss, 1, n_dof)
        # dV_all 形状: (n_gauss,)
        # 需要reshape dV_all为 (n_gauss, 1, 1)以正确广播
        dV_reshaped = dV_all.reshape(-1, 1, 1)
        vol_weighted_trace = jnp.sum(trace_B_all * dV_reshaped, axis=0) / total_vol
        
        # 为所有高斯点构建B-bar矩阵
        def build_bbar_matrix(B_std, trace_B):
            # 当前高斯点的体积部分
            B_vol_current = jnp.zeros_like(B_std)
            B_vol_current = B_vol_current.at[0:3, :].set(trace_B)
            
            # 偏差部分 + 平均体积部分
            B_dev = B_std - B_vol_current
            
            B_vol_mean = jnp.zeros_like(B_std)
            B_vol_mean = B_vol_mean.at[0:3, :].set(vol_weighted_trace)
            
            return B_dev + B_vol_mean
        
        return jax.vmap(build_bbar_matrix)(B_std_all, trace_B_all)

    def _compute_all_standard_matrices(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Return standard B matrices for all Gauss points of one C3D8 element."""
        J_all = jnp.einsum("gik,kj->gij", _DN_DXI, node_coords)
        J_inv_all = jnp.linalg.inv(J_all)
        dN_dx_all = jnp.einsum("gij,gjk->gik", J_inv_all, _DN_DXI)
        return _build_B_3d_batch(dN_dx_all, self.n_nodes)

    def sym_grad_voigt_E_all(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Return engineering-Voigt B matrices for all Gauss points.

        This is the recovery-oriented batched counterpart of
        ``sym_grad_voigt_E(..., gp_idx)`` and avoids re-evaluating the full
        B-bar volumetric average for each Gauss point separately.
        """
        if self.variant == "bbar":
            return self._compute_all_bbar_matrices_optimized(node_coords)
        return self._compute_all_standard_matrices(node_coords)

    def gauss_geometry_all(self, node_coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return Gauss coordinates and integration volumes for one element."""
        gp_coords = jnp.einsum("gn,nc->gc", _N, node_coords)
        J_all = jnp.einsum("gik,kj->gij", _DN_DXI, node_coords)
        detJ_all = jnp.linalg.det(J_all)
        return gp_coords, detJ_all * _QUAD_WEIGHTS

    def gauss_info(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _QUAD_POINTS, _QUAD_WEIGHTS
