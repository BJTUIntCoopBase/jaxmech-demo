"""Utilities for exporting `jaxmech` linear results to shakedown MAT bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import datetime as dt

import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix, issparse

from jaxmech.fem.assembly.boundary import get_dof_indices
from jaxmech.fem.element.registry import get_kernel
from jaxmech.fem.element.shape_functions import (
    hex8_shape_functions,
    quad4_shape_functions,
    tri3_shape_functions,
)
from jaxmech.fem.element.solid.c3d6 import prism6_shape_functions
from jaxmech.materials.definition import resolve_linear_material_properties
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult


_LEGACY_LOCAL_Q_FROM_JAX = np.asarray([0, 4, 3, 7, 1, 5, 2, 6], dtype=np.int32)
_JAX_LOCAL_Q_FROM_LEGACY = np.argsort(_LEGACY_LOCAL_Q_FROM_JAX)


def _unwrap_object_scalar(value):
    out = value
    while isinstance(out, np.ndarray) and out.dtype == object and out.size == 1:
        out = out.reshape(-1)[0]
    return out


def _mat_struct_to_payload(value):
    out = _unwrap_object_scalar(value)
    if issparse(out):
        return out
    if isinstance(out, dict):
        return out
    if hasattr(out, "__dict__"):
        return {k: v for k, v in out.__dict__.items() if not k.startswith("_")}
    return out


def _normalize_top_level_payload(raw: dict) -> dict:
    payload = {k: v for k, v in raw.items() if not k.startswith("__")}
    for key, value in list(payload.items()):
        payload[key] = _mat_struct_to_payload(value)
    return payload


def _reorder_c3d8_full_to_legacy_gauss_order(arr: np.ndarray, n_quads: int) -> np.ndarray:
    arr = np.asarray(arr)
    if n_quads != 8:
        return arr
    if arr.shape[0] % n_quads != 0:
        raise ValueError(f"Leading dim {arr.shape[0]} not divisible by n_quads={n_quads}.")
    n_elem = arr.shape[0] // n_quads
    reshaped = arr.reshape((n_elem, n_quads) + arr.shape[1:])
    reordered = reshaped[:, _JAX_LOCAL_Q_FROM_LEGACY, ...]
    return reordered.reshape(arr.shape)


def _shape_function_values(ele_type: str, parent_coord: np.ndarray) -> np.ndarray:
    ele_type = ele_type.upper()
    parent = jnp.asarray(parent_coord, dtype=jnp.float64)
    if ele_type in ("CPS4", "CPS4R"):
        n_vals, _ = quad4_shape_functions(parent)
        return np.asarray(n_vals, dtype=np.float64)
    if ele_type == "CPS3":
        n_vals, _ = tri3_shape_functions(parent)
        return np.asarray(n_vals, dtype=np.float64)
    if ele_type in ("C3D8", "C3D8R"):
        n_vals, _ = hex8_shape_functions(parent)
        return np.asarray(n_vals, dtype=np.float64)
    if ele_type == "C3D6":
        n_vals, _ = prism6_shape_functions(parent)
        return np.asarray(n_vals, dtype=np.float64)
    if ele_type == "C3D4":
        return np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    raise NotImplementedError(f"Unsupported shakedown export element type: {ele_type}")


def _block_stress_view(result: AnalysisResult, model: Model, block_index: int) -> np.ndarray:
    block = model.mesh.blocks[block_index]
    kernel = get_kernel(
        block.ele_type,
        variant="bbar" if block.ele_type.upper() == "C3D8" and int(result.metadata.get("use_b_ext", 1)) == 1 else None,
    )
    n_elem = block.n_elements
    n_gp = kernel.n_gauss

    if result.gauss_stress is None:
        raise ValueError("AnalysisResult.gauss_stress is required for shakedown export.")

    layout = str(result.metadata.get("gauss_layout", "element_gauss"))
    if layout == "element_gauss":
        elem_offset = sum(b.n_elements for b in model.mesh.blocks[:block_index])
        return np.asarray(result.gauss_stress[elem_offset: elem_offset + n_elem], dtype=np.float64)

    gp_offset = 0
    for prev_block in model.mesh.blocks[:block_index]:
        prev_kernel = get_kernel(
            prev_block.ele_type,
            variant="bbar" if prev_block.ele_type.upper() == "C3D8" and int(result.metadata.get("use_b_ext", 1)) == 1 else None,
        )
        gp_offset += prev_block.n_elements * prev_kernel.n_gauss
    block_flat = np.asarray(result.gauss_stress[gp_offset: gp_offset + n_elem * n_gp], dtype=np.float64)
    return block_flat.reshape(n_elem, n_gp, block_flat.shape[-1])


def export_shakedown_mat(
    model: Model,
    result: AnalysisResult,
    out_mat_path: Union[str, Path],
    ele_yield: np.ndarray,
    use_b_ext: int = 0,
    metadata: Optional[dict] = None,
    merge_existing: bool = False,
) -> Path:
    """Export a continuum shakedown bundle using the standard MAT layout."""
    family = str(model.metadata.get("family", "solid"))
    if family == "shell":
        raise NotImplementedError(
            "Shell shakedown MAT export is not implemented in this exporter. "
            "Use the dedicated shell workflow in Step 6."
        )

    out_mat_path = Path(out_mat_path)
    ndof_per_node = int(model.metadata.get("dimension", 3))
    total_dofs = model.mesh.n_nodes * ndof_per_node
    material = resolve_linear_material_properties(model)
    thickness = float(material.thickness) if ndof_per_node == 2 else 1.0

    _, _, free_dofs, _ = get_dof_indices(model, ndof_per_node)
    free_dofs = np.asarray(free_dofs, dtype=np.int32)
    dof_to_row = np.full((total_dofs,), -1, dtype=np.int32)
    dof_to_row[free_dofs] = np.arange(free_dofs.size, dtype=np.int32)

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    coords_parts: list[np.ndarray] = []
    vols_parts: list[np.ndarray] = []
    n_gp_by_block: list[int] = []

    gp_block_offset = 0
    n_str_global: Optional[int] = None

    for block_index, block in enumerate(model.mesh.blocks):
        variant = None
        if block.ele_type.upper() == "C3D8":
            variant = "bbar" if int(use_b_ext) == 1 else "standard"

        kernel = get_kernel(block.ele_type, variant=variant)
        points, weights = kernel.gauss_info()
        points = np.asarray(points, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        block_stress = _block_stress_view(result, model, block_index)
        conn_array = np.asarray(block.connectivity, dtype=np.int32)
        coords_batch_j = jnp.asarray(model.mesh.nodes[conn_array], dtype=jnp.float64)

        n_elem = block.n_elements
        n_gp = kernel.n_gauss
        n_str = kernel.n_str
        n_gp_by_block.append(n_gp)
        if n_str_global is None:
            n_str_global = n_str
        elif n_str_global != n_str:
            raise ValueError("All blocks in one shakedown export must share the same n_str.")

        block_sigma_legacy = np.zeros((n_elem, n_gp, n_str), dtype=np.float64)
        block_coords_legacy = np.zeros((n_elem, n_gp, model.mesh.n_dim), dtype=np.float64)
        block_vols_legacy = np.zeros((n_elem, n_gp), dtype=np.float64)

        coords_batch_np = None
        vols_batch_np = None
        weighted_bt_batch_np = None
        kernel_gauss_geometry_all = getattr(kernel, "gauss_geometry_all", None)
        kernel_b_all = getattr(kernel, "sym_grad_voigt_E_all", None)
        if kernel_gauss_geometry_all is not None:
            jax_mod = __import__("jax")
            coords_batch_jax, vols_batch_jax = jax_mod.vmap(kernel_gauss_geometry_all)(coords_batch_j)
            if ndof_per_node == 2:
                vols_batch_jax = vols_batch_jax * thickness
            coords_batch_np = np.asarray(coords_batch_jax, dtype=np.float64)
            vols_batch_np = np.asarray(vols_batch_jax, dtype=np.float64)
        if kernel_b_all is not None and vols_batch_np is not None:
            jax_mod = __import__("jax")
            b_batch_jax = jax_mod.vmap(kernel_b_all)(coords_batch_j)
            weighted_bt_batch_np = np.asarray(
                jnp.swapaxes(b_batch_jax, -1, -2) * jnp.asarray(vols_batch_np)[..., None, None],
                dtype=np.float64,
            )

        if block.ele_type.upper() == "C3D8" and n_gp == 8:
            block_sigma_legacy = np.asarray(block_stress[:, _JAX_LOCAL_Q_FROM_LEGACY, :], dtype=np.float64)
            if coords_batch_np is not None:
                block_coords_legacy = coords_batch_np[:, _JAX_LOCAL_Q_FROM_LEGACY, :]
            if vols_batch_np is not None:
                block_vols_legacy = vols_batch_np[:, _JAX_LOCAL_Q_FROM_LEGACY]
            if weighted_bt_batch_np is not None:
                weighted_bt_batch_np = weighted_bt_batch_np[:, _JAX_LOCAL_Q_FROM_LEGACY, :, :]
        else:
            block_sigma_legacy = np.asarray(block_stress, dtype=np.float64)
            if coords_batch_np is not None:
                block_coords_legacy = coords_batch_np
            if vols_batch_np is not None:
                block_vols_legacy = vols_batch_np

        elem_dofs_all = (
            conn_array[:, :, None] * ndof_per_node
            + np.arange(ndof_per_node, dtype=np.int32)[None, None, :]
        ).reshape(n_elem, kernel.n_nodes * ndof_per_node)
        row_inds_all = dof_to_row[elem_dofs_all]
        active_all = row_inds_all >= 0

        if weighted_bt_batch_np is not None:
            gp_ids = gp_block_offset + np.arange(n_elem * n_gp, dtype=np.int32).reshape(n_elem, n_gp)
            row_grid = np.broadcast_to(
                row_inds_all[:, None, :, None],
                (n_elem, n_gp, kernel.n_nodes * ndof_per_node, n_str),
            )
            col_grid = np.broadcast_to(
                gp_ids[:, :, None, None] * n_str + np.arange(n_str, dtype=np.int32)[None, None, None, :],
                (n_elem, n_gp, kernel.n_nodes * ndof_per_node, n_str),
            )
            mask = np.broadcast_to(
                active_all[:, None, :, None],
                (n_elem, n_gp, kernel.n_nodes * ndof_per_node, n_str),
            ) & (np.abs(weighted_bt_batch_np) > 0.0)

            if np.any(mask):
                row_parts.append(row_grid[mask])
                col_parts.append(col_grid[mask])
                data_parts.append(weighted_bt_batch_np[mask])

            sigma_parts.append(block_sigma_legacy.reshape(-1, n_str))
            coords_parts.append(block_coords_legacy.reshape(-1, model.mesh.n_dim))
            vols_parts.append(block_vols_legacy.reshape(-1, 1))
            gp_block_offset += n_elem * n_gp
            continue

        for elem_id, conn in enumerate(conn_array):
            elem_dofs = elem_dofs_all[elem_id]
            row_inds = row_inds_all[elem_id]
            active = active_all[elem_id]
            node_coords = np.asarray(model.mesh.nodes[conn], dtype=np.float64)

            for gp_idx, (parent_coord, weight) in enumerate(zip(points, weights)):
                b_mat = np.asarray(kernel.sym_grad_voigt_E(jnp.asarray(node_coords, dtype=jnp.float64), gp_idx), dtype=np.float64)
                _, _, det_j = kernel.jacobian(jnp.asarray(node_coords, dtype=jnp.float64), gp_idx)
                det_j = float(np.asarray(det_j))
                local_legacy_idx = int(_LEGACY_LOCAL_Q_FROM_JAX[gp_idx]) if block.ele_type.upper() == "C3D8" and n_gp == 8 else gp_idx
                gp_id = gp_block_offset + elem_id * n_gp + local_legacy_idx

                jxw = det_j * float(weight) * thickness
                shape_vals = _shape_function_values(block.ele_type, parent_coord)
                gp_coord = shape_vals @ node_coords

                block_sigma_legacy[elem_id, local_legacy_idx, :] = block_stress[elem_id, gp_idx, :]
                block_coords_legacy[elem_id, local_legacy_idx, :] = gp_coord
                block_vols_legacy[elem_id, local_legacy_idx] = jxw

                if np.any(active):
                    weighted_bt = (b_mat.T * jxw)[active]
                    rr = np.repeat(row_inds[active], n_str)
                    cc = np.tile(np.arange(gp_id * n_str, (gp_id + 1) * n_str, dtype=np.int32), int(np.sum(active)))
                    vv = weighted_bt.reshape(-1)
                    nz = np.abs(vv) > 0.0
                    if np.any(nz):
                        row_parts.append(rr[nz])
                        col_parts.append(cc[nz])
                        data_parts.append(vv[nz])

        sigma_parts.append(block_sigma_legacy.reshape(-1, n_str))
        coords_parts.append(block_coords_legacy.reshape(-1, model.mesh.n_dim))
        vols_parts.append(block_vols_legacy.reshape(-1, 1))
        gp_block_offset += n_elem * n_gp

    n_gauss_total = gp_block_offset
    n_str_out = int(n_str_global or 0)
    ele_yield = np.asarray(ele_yield, dtype=np.float64).reshape(-1)
    if ele_yield.size != n_gauss_total:
        raise ValueError(
            f"ele_yield size mismatch: got {ele_yield.size}, expected {n_gauss_total} gauss points."
        )

    if data_parts:
        c_sparse = coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
            shape=(free_dofs.size, n_gauss_total * n_str_out),
            dtype=np.float64,
        ).tocsr()
    else:
        c_sparse = coo_matrix((free_dofs.size, n_gauss_total * n_str_out), dtype=np.float64).tocsr()

    unique_n_gp = sorted(set(n_gp_by_block))
    mat_dict = {
        "gauss_coords": np.concatenate(coords_parts, axis=0),
        "gauss_vols": np.concatenate(vols_parts, axis=0),
        "n_gauss": np.asarray([[int(n_gauss_total)]], dtype=np.int64),
        "n_gauss_per_ele": np.asarray([[int(unique_n_gp[0]) if len(unique_n_gp) == 1 else -1]], dtype=np.int64),
        "free_dofs": np.asarray(free_dofs, dtype=np.int64).reshape(1, -1),
        "sigma_E": np.concatenate(sigma_parts, axis=0),
        "ele_yield": ele_yield.reshape(-1, 1),
        "n_str": np.asarray([[n_str_out]], dtype=np.int64),
        "use_b_ext": np.asarray([[int(use_b_ext)]], dtype=np.int64),
        "C_sparse": c_sparse,
        "created_at": np.asarray([dt.datetime.now().isoformat()]),
    }
    if len(unique_n_gp) > 1:
        mat_dict["n_gauss_per_ele_by_type"] = np.asarray(n_gp_by_block, dtype=np.int32).reshape(-1, 1)

    merged_meta = dict(metadata or {})
    merged_meta.setdefault("source", "jaxmech.export_shakedown_mat")
    merged_meta.setdefault("analysis_type", result.metadata.get("analysis_type", "linear_static"))
    for key, value in merged_meta.items():
        mat_dict[f"meta_{key}"] = np.asarray([value]) if isinstance(value, str) else np.asarray(value)

    if merge_existing and Path(out_mat_path).is_file():
        raw = sio.loadmat(str(out_mat_path), squeeze_me=False, struct_as_record=False)
        merged = _normalize_top_level_payload(raw)
        merged.update(mat_dict)
        mat_dict = merged

    sio.savemat(out_mat_path, mat_dict, do_compression=True)
    return out_mat_path
