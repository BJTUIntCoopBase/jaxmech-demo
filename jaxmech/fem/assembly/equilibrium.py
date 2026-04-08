"""Assemble equilibrium operators from element kinematics."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import coo_matrix

from jaxmech.fem.assembly.boundary import get_dof_indices
from jaxmech.fem.element.registry import get_kernel
from jaxmech.model.model import Model


def _block_weighted_bt_batch(kernel, coords_batch: jnp.ndarray, thickness: float) -> jnp.ndarray:
    """Return ``(n_elem, n_gp, n_elem_dof, n_str)`` weighted ``B^T`` blocks."""
    if hasattr(kernel, "sym_grad_voigt_E_all") and hasattr(kernel, "gauss_geometry_all"):
        b_batch = jax.vmap(kernel.sym_grad_voigt_E_all)(coords_batch)
        _, dvol_batch = jax.vmap(kernel.gauss_geometry_all)(coords_batch)
        if int(kernel.n_dim) == 2:
            dvol_batch = dvol_batch * thickness
        return jnp.swapaxes(b_batch, -1, -2) * dvol_batch[..., None, None]

    _, weights = kernel.gauss_info()
    weights = jnp.asarray(weights, dtype=jnp.float64)

    def per_element(node_coords: jnp.ndarray) -> jnp.ndarray:
        def per_gp(gp_idx: int) -> jnp.ndarray:
            b_mat = kernel.sym_grad_voigt_E(node_coords, gp_idx)
            _, _, det_j = kernel.jacobian(node_coords, gp_idx)
            jxw = det_j * weights[gp_idx] * thickness
            return b_mat.T * jxw

        return jax.vmap(per_gp)(jnp.arange(kernel.n_gauss))

    return jax.vmap(per_element)(coords_batch)


def build_solid_c_sparse(
    model: Model,
    *,
    ndof_per_node: int,
    use_b_ext: int = 1,
    thickness: float = 1.0,
) -> tuple[coo_matrix, np.ndarray, int]:
    """Assemble the solid equilibrium matrix ``C_sparse``.

    Returns
    -------
    tuple
        ``(C_sparse, free_dofs, n_str)`` where ``C_sparse`` maps flattened
        Gauss stresses to free-DOF internal forces.
    """
    if str(model.metadata.get("family", "solid")) != "solid":
        raise NotImplementedError("build_solid_c_sparse currently supports solid models only.")

    _, _, free_dofs, _ = get_dof_indices(model, ndof_per_node)
    free_dofs = np.asarray(free_dofs, dtype=np.int32)
    total_dofs = model.mesh.n_nodes * ndof_per_node
    dof_to_row = np.full((total_dofs,), -1, dtype=np.int32)
    dof_to_row[free_dofs] = np.arange(free_dofs.size, dtype=np.int32)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    gp_offset = 0
    n_str_global: int | None = None

    for block in model.mesh.blocks:
        variant = None
        if block.ele_type.upper() == "C3D8":
            variant = "bbar" if int(use_b_ext) == 1 else "standard"

        kernel = get_kernel(block.ele_type, variant=variant)
        n_elem = block.n_elements
        n_gp = kernel.n_gauss
        n_str = kernel.n_str
        n_elem_dof = kernel.n_nodes * ndof_per_node

        if n_str_global is None:
            n_str_global = int(n_str)
        elif int(n_str_global) != int(n_str):
            raise ValueError("All blocks in one solid C_sparse must share the same n_str.")

        conn_array = np.asarray(block.connectivity, dtype=np.int32)
        coords_batch = jnp.asarray(model.mesh.nodes[conn_array], dtype=jnp.float64)
        weighted_bt_batch = np.asarray(_block_weighted_bt_batch(kernel, coords_batch, thickness), dtype=np.float64)

        elem_dofs = (
            conn_array[:, :, None] * ndof_per_node
            + np.arange(ndof_per_node, dtype=np.int32)[None, None, :]
        ).reshape(n_elem, n_elem_dof)
        row_idx = dof_to_row[elem_dofs]
        active = row_idx >= 0

        gp_ids = gp_offset + np.arange(n_elem * n_gp, dtype=np.int32).reshape(n_elem, n_gp)
        row_grid = np.broadcast_to(row_idx[:, None, :, None], (n_elem, n_gp, n_elem_dof, n_str))
        col_grid = np.broadcast_to(
            gp_ids[:, :, None, None] * n_str + np.arange(n_str, dtype=np.int32)[None, None, None, :],
            (n_elem, n_gp, n_elem_dof, n_str),
        )
        mask = np.broadcast_to(active[:, None, :, None], (n_elem, n_gp, n_elem_dof, n_str)) & (np.abs(weighted_bt_batch) > 0.0)

        if np.any(mask):
            row_parts.append(row_grid[mask])
            col_parts.append(col_grid[mask])
            data_parts.append(weighted_bt_batch[mask])

        gp_offset += n_elem * n_gp

    n_str_out = int(n_str_global or 0)
    if data_parts:
        c_sparse = coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(free_dofs.size, gp_offset * n_str_out),
            dtype=np.float64,
        ).tocsr()
    else:
        c_sparse = coo_matrix((free_dofs.size, gp_offset * n_str_out), dtype=np.float64).tocsr()

    return c_sparse, free_dofs, n_str_out
