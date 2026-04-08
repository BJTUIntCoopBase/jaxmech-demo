"""
jaxmech.fem.assembly.assembler — Sparse stiffness assembly & field recovery.

After the R1 refactoring, element kernels provide only geometric operators
(B matrices, Jacobians).  This module combines them with material constitutive
matrices to build global stiffness, compute internal forces, and recover
Gauss-point strain/stress fields.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np

from jaxmech.fem.mesh import Mesh, ElementBlock
from jaxmech.fem.element.registry import get_kernel

from jaxmech.materials.elastic import get_d_matrix as _get_d_matrix


# ======================================================================
#  Element-level computations (combined kernel geometry + material D)
# ======================================================================

def _is_shell(kernel) -> bool:
    """Check if this is a shell element."""
    return kernel.ele_type in ("STRI3", "S4", "S4R")


def _solid_batched_b_and_dvol(kernel, node_coords: jnp.ndarray,
                              thickness: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return all solid B matrices and integration volumes for one element."""
    B_all = kernel.sym_grad_voigt_E_all(node_coords)

    kernel_gauss_geometry_all = getattr(kernel, 'gauss_geometry_all', None)
    if kernel_gauss_geometry_all is not None:
        _, dvol_all = kernel_gauss_geometry_all(node_coords)
    else:
        _, weights = kernel.gauss_info()
        weights = jnp.asarray(weights, dtype=jnp.float64)

        def _detj_times_w(gp_idx):
            _, _, detJ = kernel.jacobian(node_coords, gp_idx)
            return detJ * weights[gp_idx]

        dvol_all = jax.vmap(_detj_times_w)(jnp.arange(kernel.n_gauss, dtype=jnp.int32))

    if int(kernel.n_dim) == 2:
        dvol_all = dvol_all * thickness

    return B_all, dvol_all


def _solid_batched_b_and_dvol_batch(kernel, coords_batch: jnp.ndarray,
                                    thickness: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return batched solid B matrices and integration volumes for a block."""
    return jax.vmap(lambda node_coords: _solid_batched_b_and_dvol(kernel, node_coords, thickness))(coords_batch)


def _element_stiffness_solid(kernel, node_coords: jnp.ndarray,
                              D: jnp.ndarray, thickness: float) -> jnp.ndarray:
    """K_e = Σ_q B_q^T D B_q detJ w_q [* thickness for 2D]

    Works for all solid / 2D plane elements.
    """
    if hasattr(kernel, 'sym_grad_voigt_E_all'):
        B_all, dvol_all = _solid_batched_b_and_dvol(kernel, node_coords, thickness)
        return jnp.einsum('gsi,st,gtj,g->ij', B_all, D, B_all, dvol_all)

    n_dof = kernel.n_nodes * kernel.n_dim
    _, weights = kernel.gauss_info()

    def iter_gp(gp_idx, K_acc):
        B = kernel.sym_grad_voigt_E(node_coords, gp_idx)
        _, _, detJ = kernel.jacobian(node_coords, gp_idx)
        w = weights[gp_idx]
        K_gp = jnp.dot(B.T, jnp.dot(D, B)) * detJ * w * thickness
        return K_acc + K_gp

    K_local = jnp.zeros((n_dof, n_dof), dtype=jnp.float64)
    K_local = jax.lax.fori_loop(0, kernel.n_gauss, iter_gp, K_local)
    return K_local


def _element_stiffness_shell(kernel, node_coords: jnp.ndarray,
                              D: jnp.ndarray) -> jnp.ndarray:
    """Element stiffness for shell, using JaxSSO optimized path.

    Falls back to B^T D B integration if the kernel doesn't have
    the optimized ``element_stiffness_global`` method.
    """
    E = float(D[0, 0] * (1.0 - D[0, 1]**2 / D[0, 0]**2))  # recover E from D
    # Since D = Cm/t * t = Cm, and Cm = E/(1-nu^2) * [...], recovering E/nu
    # is fragile. Instead, use the `element_stiffness_global` shortcut:
    if hasattr(kernel, 'element_stiffness_global'):
        # Extract E, nu, thickness from the generalized D
        # Cm = D[0:3, 0:3], and Cm = E*t/(1-nu^2) * [...]
        # Cb = D[3:6, 3:6], and Cb = E*t^3/(12*(1-nu^2)) * [...]
        # ratio Cb[0,0] / Cm[0,0] = t^2/12 => t = sqrt(12 * Cb[0,0]/Cm[0,0])
        Cm00 = D[0, 0]
        Cb00 = D[3, 3]
        t = jnp.sqrt(12.0 * Cb00 / Cm00)
        nu_est = D[0, 1] / Cm00 * (1.0 - D[0, 1] / Cm00)
        # Actually, easier to just pass-through from material_props.
        # This method is called from _element_stiffness which receives D not props.
        # We'll handle this at the caller level.
        pass

    # General path: B^T D B in local, then transform to global
    T = kernel.transformation_matrix(node_coords)
    _, weights = kernel.gauss_info()
    n_dof = kernel.n_nodes * kernel.n_dim

    def iter_gp(gp_idx, K_acc):
        B_local = kernel.sym_grad_voigt_E(node_coords, gp_idx)
        _, _, detJ = kernel.jacobian(node_coords, gp_idx)
        w = weights[gp_idx]
        K_gp = jnp.dot(B_local.T, jnp.dot(D, B_local)) * detJ * w
        return K_acc + K_gp

    K_local = jnp.zeros((n_dof, n_dof), dtype=jnp.float64)
    K_local = jax.lax.fori_loop(0, kernel.n_gauss, iter_gp, K_local)

    # Transform to global: K_global = T^T @ K_local @ T
    K_global = jnp.dot(T.T, jnp.dot(K_local, T))
    return K_global


def _element_stiffness(kernel, node_coords: jnp.ndarray,
                        material_props: dict) -> jnp.ndarray:
    """Unified element stiffness dispatch."""
    if _is_shell(kernel) and hasattr(kernel, 'element_stiffness_global'):
        # Use JaxSSO optimized path directly
        E = material_props["E"]
        nu = material_props["nu"]
        thickness = material_props.get("thickness", 1.0)
        return kernel.element_stiffness_global(node_coords, E, nu, thickness)

    D = _get_d_matrix(material_props, kernel)
    if _is_shell(kernel):
        return _element_stiffness_shell(kernel, node_coords, D)

    thickness = material_props.get("thickness", 1.0)
    return _element_stiffness_solid(kernel, node_coords, D, thickness)


def _element_recover(kernel, node_coords: jnp.ndarray,
                      u_local: jnp.ndarray,
                      material_props: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Recover strain/stress at all Gauss points for one element.

    Returns (n_gauss, n_str) arrays for strain and stress.
    """
    D = _get_d_matrix(material_props, kernel)

    if _is_shell(kernel):
        # Transform global displacements to local
        T = kernel.transformation_matrix(node_coords)
        u_loc = jnp.dot(T, u_local)

        def scan_fn(_, gp_idx):
            B_local = kernel.sym_grad_voigt_E(node_coords, gp_idx)
            strain = jnp.dot(B_local, u_loc)
            stress = jnp.dot(D, strain)
            return None, (strain, stress)

        _, (strains, stresses) = jax.lax.scan(
            scan_fn, None, jnp.arange(kernel.n_gauss)
        )
        return strains, stresses

    # Solid / 2D path
    def scan_fn(_, gp_idx):
        B = kernel.sym_grad_voigt_E(node_coords, gp_idx)
        strain = jnp.dot(B, u_local)
        stress = jnp.dot(D, strain)
        return None, (strain, stress)

    _, (strains, stresses) = jax.lax.scan(
        scan_fn, None, jnp.arange(kernel.n_gauss)
    )
    return strains, stresses


def _element_recover_optimized(kernel, node_coords: jnp.ndarray,
                              u_local: jnp.ndarray,
                              D: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """🚀 Optimized strain/stress recovery with pre-computed D matrix.
    
    Key optimizations:
    - D matrix is pre-computed (no repeated calls to _get_d_matrix)
    - Uses vmap instead of scan for better vectorization
    - Special handling for C3D8 B-bar to avoid O(n²) complexity
    
    🔧 Includes fallback to ensure robustness
    """
    # 🔧 安全检查：验证输入参数
    try:
        if _is_shell(kernel):
            # Transform global displacements to local
            T = kernel.transformation_matrix(node_coords)
            u_loc = jnp.dot(T, u_local)

            def compute_at_gp(gp_idx):
                B_local = kernel.sym_grad_voigt_E(node_coords, gp_idx)
                strain = jnp.dot(B_local, u_loc)
                stress = jnp.dot(D, strain)
                return strain, stress

            # 🚀 优化2: 使用vmap替代scan (更好的并行化)
            strains, stresses = jax.vmap(compute_at_gp)(jnp.arange(kernel.n_gauss))
            return strains, stresses

        if hasattr(kernel, 'sym_grad_voigt_E_all'):
            B_all = kernel.sym_grad_voigt_E_all(node_coords)
            strains = jnp.einsum('gij,j->gi', B_all, u_local)
            stresses = jnp.einsum('ij,gj->gi', D, strains)
            return strains, stresses

        # Solid / 2D path - 标准vmap优化
        def compute_at_gp(gp_idx):
            B = kernel.sym_grad_voigt_E(node_coords, gp_idx)
            strain = jnp.dot(B, u_local)
            stress = jnp.dot(D, strain)
            return strain, stress

        # 🚀 优化2: 使用vmap替代scan
        strains, stresses = jax.vmap(compute_at_gp)(jnp.arange(kernel.n_gauss))
        return strains, stresses
        
    except Exception as e:
        # 🔧 如果优化版本失败，使用保守的scan方法但保留预计算的D矩阵
        print(f"  [Recovery] 优化计算失败，使用scan方法: {str(e)}", flush=True)
        
        if _is_shell(kernel):
            T = kernel.transformation_matrix(node_coords)
            u_loc = jnp.dot(T, u_local)
            
            def scan_fn(_, gp_idx):
                B_local = kernel.sym_grad_voigt_E(node_coords, gp_idx)
                strain = jnp.dot(B_local, u_loc)
                stress = jnp.dot(D, strain)
                return None, (strain, stress)
                
            _, (strains, stresses) = jax.lax.scan(
                scan_fn, None, jnp.arange(kernel.n_gauss)
            )
            return strains, stresses
        else:
            def scan_fn(_, gp_idx):
                B = kernel.sym_grad_voigt_E(node_coords, gp_idx)
                strain = jnp.dot(B, u_local)
                stress = jnp.dot(D, strain)
                return None, (strain, stress)
                
            _, (strains, stresses) = jax.lax.scan(
                scan_fn, None, jnp.arange(kernel.n_gauss)
            )
            return strains, stresses


# ======================================================================
#  Global assembly (public API — unchanged signatures)
# ======================================================================

def _block_stiffness_vmap(coords_batch: jnp.ndarray, material_props: dict, kernel) -> jnp.ndarray:
    """Evaluate stiffness matrix for a batch of elements using vmap."""
    if not _is_shell(kernel):
        D = _get_d_matrix(material_props, kernel)
        thickness = material_props.get("thickness", 1.0)
        B_batch, dvol_batch = _solid_batched_b_and_dvol_batch(kernel, coords_batch, thickness)
        return jnp.einsum('egsi,st,egtj,eg->eij', B_batch, D, B_batch, dvol_batch)

    def single_k(node_coords):
        return _element_stiffness(kernel, node_coords, material_props)

    return jax.vmap(single_k)(coords_batch)


def build_global_stiffness(mesh: Mesh, material_props: dict, ndof_per_node: int = None) -> sparse.BCOO:
    """Assemble the global stiffness matrix in JAX BCOO format.

    Parameters
    ----------
    mesh : Mesh
    material_props : dict
        Material properties (E, nu, thickness, etc.).
    ndof_per_node : int, optional
        If None, inferred from the elements.

    Returns
    -------
    K_global : JAX sparse.BCOO
    """
    if not mesh.blocks:
        raise ValueError("Mesh contains no element blocks.")

    if ndof_per_node is None:
        ndof_per_node = 3
        for block in mesh.blocks:
            if "S" in block.ele_type.upper() and not block.ele_type.upper().startswith("CPS"):
                ndof_per_node = 6
                break

    n_nodes = mesh.n_nodes
    total_dofs = n_nodes * ndof_per_node

    allOriginValues = []
    allRowIndices = []
    allColIndices = []

    for block in mesh.blocks:
        # Resolve variant from material_props
        variant = None
        if block.ele_type.upper() == "C3D8":
            use_b_ext = material_props.get("use_b_ext", 1)
            variant = "bbar" if int(use_b_ext) == 1 else "standard"

        kernel = get_kernel(block.ele_type, variant=variant)
        n_elem = block.n_elements
        n_elem_nodes = kernel.n_nodes
        n_elem_dofs = n_elem_nodes * ndof_per_node

        # 1. Gather coordinates for this block
        conn = jnp.array(block.connectivity, dtype=jnp.int32)
        nodes_batch = jnp.array(mesh.nodes, dtype=jnp.float64)[conn]

        # 2. Vmap stiffness evaluation
        K_local_batch = _block_stiffness_vmap(nodes_batch, material_props, kernel)

        # 3. Build global DOF indices
        node_dof_base = conn * ndof_per_node
        dof_offsets = jnp.arange(ndof_per_node, dtype=jnp.int32)
        block_dofs = node_dof_base[..., None] + dof_offsets
        block_dofs = block_dofs.reshape(n_elem, n_elem_dofs)

        # 4. COO entries
        rows = jnp.repeat(block_dofs, n_elem_dofs, axis=1).reshape(n_elem, n_elem_dofs, n_elem_dofs)
        cols = jnp.tile(block_dofs[:, None, :], (1, n_elem_dofs, 1))

        allOriginValues.append(K_local_batch.flatten())
        allRowIndices.append(rows.flatten())
        allColIndices.append(cols.flatten())

    data = jnp.concatenate(allOriginValues)
    rows = jnp.concatenate(allRowIndices)
    cols = jnp.concatenate(allColIndices)

    indices = jnp.stack([rows, cols], axis=-1)
    K = sparse.BCOO((data, indices), shape=(total_dofs, total_dofs))
    K = K.sort_indices()
    K = K.sum_duplicates()
    return K


def _block_recover_vmap(coords_batch: jnp.ndarray, u_local_batch: jnp.ndarray,
                         material_props: dict, kernel) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate strains and stresses for a batch of elements."""
    def single_recover(node_coords, u_local):
        return _element_recover(kernel, node_coords, u_local, material_props)

    return jax.vmap(single_recover)(coords_batch, u_local_batch)


def _block_recover_vmap_optimized(coords_batch: jnp.ndarray, u_local_batch: jnp.ndarray,
                                 D: jnp.ndarray, kernel) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """🚀 Optimized version: Pre-computed D matrix, vectorized operations."""
    if hasattr(kernel, 'sym_grad_voigt_E_all'):
        B_batch = jax.vmap(kernel.sym_grad_voigt_E_all)(coords_batch)
        strains = jnp.einsum('egij,ej->egi', B_batch, u_local_batch)
        stresses = jnp.einsum('ij,egj->egi', D, strains)
        return strains, stresses

    def single_recover_optimized(node_coords, u_local):
        return _element_recover_optimized(kernel, node_coords, u_local, D)

    return jax.vmap(single_recover_optimized)(coords_batch, u_local_batch)


def recover_all_fields(mesh: Mesh, u_global: jnp.ndarray, material_props: dict,
                        ndof_per_node: int = None) -> dict:
    """Recover strains and stresses at all integration points for all blocks.

    Returns
    -------
    dict : mapping block_id -> {'strain': ndarray, 'stress': ndarray}
           Arrays have shape (n_elem, n_gauss, n_str)
    """
    if ndof_per_node is None:
        ndof_per_node = 3
        for block in mesh.blocks:
            if "S" in block.ele_type.upper() and not block.ele_type.upper().startswith("CPS"):
                ndof_per_node = 6
                break

    results = {}
    for block in mesh.blocks:
        variant = None
        if block.ele_type.upper() == "C3D8":
            use_b_ext = material_props.get("use_b_ext", 1)
            variant = "bbar" if int(use_b_ext) == 1 else "standard"

        kernel = get_kernel(block.ele_type, variant=variant)
        
        # 🚀 优化1: 预计算D矩阵 (块级别共享)
        D = _get_d_matrix(material_props, kernel)
        
        conn = jnp.array(block.connectivity, dtype=jnp.int32)
        nodes_batch = jnp.array(mesh.nodes, dtype=jnp.float64)[conn]

        node_dof_base = conn * ndof_per_node
        dof_offsets = jnp.arange(ndof_per_node, dtype=jnp.int32)
        block_dofs = node_dof_base[..., None] + dof_offsets
        block_dofs = block_dofs.reshape(block.n_elements, kernel.n_nodes * ndof_per_node)

        u_local_batch = u_global[block_dofs]

        # 🔧 安全回退：尝试优化版本，失败时使用原始版本
        try:
            strains, stresses = _block_recover_vmap_optimized(nodes_batch, u_local_batch, D, kernel)
        except Exception as e:
            print(f"  [Recovery] 优化版本失败，回退到原始版本: {str(e)}", flush=True)
            strains, stresses = _block_recover_vmap(nodes_batch, u_local_batch, material_props, kernel)
        results[block.block_id] = {
            "strain": strains,
            "stress": stresses
        }

    return results


def build_internal_force(mesh: Mesh, u_global: jnp.ndarray, material_props: dict,
                          ndof_per_node: int = None) -> jnp.ndarray:
    """Compute global internal force vector by assembling local element forces.

    Returns
    -------
    f_int_global : jnp.ndarray of shape (n_nodes * ndof_per_node,)
    """
    if ndof_per_node is None:
        ndof_per_node = 3
        for block in mesh.blocks:
            if "S" in block.ele_type.upper() and not block.ele_type.upper().startswith("CPS"):
                ndof_per_node = 6
                break

    n_nodes = mesh.n_nodes
    total_dofs = n_nodes * ndof_per_node
    f_int_global = jnp.zeros(total_dofs, dtype=jnp.float64)

    for block in mesh.blocks:
        variant = None
        if block.ele_type.upper() == "C3D8":
            use_b_ext = material_props.get("use_b_ext", 1)
            variant = "bbar" if int(use_b_ext) == 1 else "standard"

        kernel = get_kernel(block.ele_type, variant=variant)
        D = _get_d_matrix(material_props, kernel)
        thickness = material_props.get("thickness", 1.0)
        _, weights = kernel.gauss_info()

        conn = jnp.array(block.connectivity, dtype=jnp.int32)
        nodes_batch = jnp.array(mesh.nodes, dtype=jnp.float64)[conn]

        node_dof_base = conn * ndof_per_node
        dof_offsets = jnp.arange(ndof_per_node, dtype=jnp.int32)
        block_dofs = node_dof_base[..., None] + dof_offsets
        block_dofs = block_dofs.reshape(block.n_elements, kernel.n_nodes * ndof_per_node)

        u_local_batch = u_global[block_dofs]

        if _is_shell(kernel):
            def single_f_int(node_coords, u_elem):
                T = kernel.transformation_matrix(node_coords)
                u_loc = jnp.dot(T, u_elem)
                f_loc = jnp.zeros_like(u_loc)
                def iter_gp(gp_idx, f_acc):
                    B_local = kernel.sym_grad_voigt_E(node_coords, gp_idx)
                    _, _, detJ = kernel.jacobian(node_coords, gp_idx)
                    w = weights[gp_idx]
                    strain = jnp.dot(B_local, u_loc)
                    stress = jnp.dot(D, strain)
                    f_gp = jnp.dot(B_local.T, stress) * detJ * w
                    return f_acc + f_gp
                f_loc = jax.lax.fori_loop(0, kernel.n_gauss, iter_gp, f_loc)
                # Transform back to global
                return jnp.dot(T.T, f_loc)
            f_local_batch = jax.vmap(single_f_int)(nodes_batch, u_local_batch)
        else:
            B_batch, dvol_batch = _solid_batched_b_and_dvol_batch(kernel, nodes_batch, thickness)
            strains = jnp.einsum('egsi,ei->egs', B_batch, u_local_batch)
            stresses = jnp.einsum('st,egt->egs', D, strains)
            f_local_batch = jnp.einsum('egsi,egs,eg->ei', B_batch, stresses, dvol_batch)

        f_int_global = jnp.zeros_like(f_int_global).at[block_dofs].add(f_local_batch) + f_int_global

    return f_int_global
