"""Single-step linear solver used by the incremental analysis driver."""

from __future__ import annotations

import time
from typing import Optional

import jax
import numpy as np
import scipy.sparse as sp

import jax.numpy as jnp

from jaxmech.model.model import Model
from jaxmech.materials.definition import LinearMaterialProperties
from jaxmech.model.result import AnalysisResult, Operators
from jaxmech.fem.assembly.assembler import build_global_stiffness, recover_all_fields
from jaxmech.fem.assembly.boundary import build_load_vector, get_dof_indices
from jaxmech.fem.assembly.equilibrium import build_solid_c_sparse
from jaxmech.fem.element.registry import get_kernel
from jaxmech.fem.element.shape_functions import (
    hex8_shape_functions,
    quad4_shape_functions,
    tri3_shape_functions,
)
from jaxmech.fem.element.solid.c3d6 import prism6_shape_functions
from jaxmech.fem.solver.linear import linear_solve


def _collect_gauss_fields(model: Model, recover_results: dict):
    """Collect Gauss fields into a consistent non-ragged layout.

    For homogeneous meshes where all blocks share the same number of Gauss
    points, returns arrays of shape ``(n_elem, n_gp, n_str)``.

    For mixed meshes with different numbers of Gauss points per element,
    returns flattened arrays of shape ``(n_gauss_total, n_str)`` plus
    ``n_gauss_per_elem`` so downstream modules can reconstruct per-element
    segments without introducing NaNs.
    """
    block_strains: list[np.ndarray] = []
    block_stresses: list[np.ndarray] = []
    n_gauss_per_elem_parts: list[np.ndarray] = []

    for block in model.mesh.blocks:
        strain = np.asarray(recover_results[block.block_id]["strain"])
        stress = np.asarray(recover_results[block.block_id]["stress"])
        block_strains.append(strain)
        block_stresses.append(stress)
        n_gauss_per_elem_parts.append(
            np.full(block.n_elements, strain.shape[1], dtype=np.int32)
        )

    if not block_stresses:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, empty, np.empty((0,), dtype=np.int32), "flat"

    n_gauss_per_elem = np.concatenate(n_gauss_per_elem_parts, axis=0)
    unique_gp = {arr.shape[1] for arr in block_stresses}

    if len(unique_gp) == 1:
        gauss_strain = np.concatenate(block_strains, axis=0)
        gauss_stress = np.concatenate(block_stresses, axis=0)
        return gauss_strain, gauss_stress, n_gauss_per_elem, "element_gauss"

    flat_strains = [arr.reshape(-1, arr.shape[-1]) for arr in block_strains]
    flat_stresses = [arr.reshape(-1, arr.shape[-1]) for arr in block_stresses]
    gauss_strain = np.concatenate(flat_strains, axis=0)
    gauss_stress = np.concatenate(flat_stresses, axis=0)
    return gauss_strain, gauss_stress, n_gauss_per_elem, "flat"


def _shape_values(ele_type: str, gp_local: np.ndarray) -> np.ndarray:
    ele = str(ele_type).upper()
    if ele in {"CPS4", "CPE4", "CPS4R", "CPE4R"}:
        return np.asarray(quad4_shape_functions(jnp.asarray(gp_local, dtype=jnp.float64))[0], dtype=np.float64)
    if ele in {"CPS3", "CPE3"}:
        return np.asarray(tri3_shape_functions(jnp.asarray(gp_local, dtype=jnp.float64))[0], dtype=np.float64)
    if ele in {"C3D8", "C3D8R"}:
        return np.asarray(hex8_shape_functions(jnp.asarray(gp_local, dtype=jnp.float64))[0], dtype=np.float64)
    if ele == "C3D6":
        return np.asarray(prism6_shape_functions(jnp.asarray(gp_local, dtype=jnp.float64))[0], dtype=np.float64)
    if ele == "C3D4":
        xi, eta, zeta = [float(v) for v in np.asarray(gp_local, dtype=np.float64).reshape(-1)]
        return np.asarray([1.0 - xi - eta - zeta, xi, eta, zeta], dtype=np.float64)
    raise NotImplementedError(f"Gauss coordinate recovery is not implemented for element type {ele!r}.")


def _collect_gauss_geometry(model: Model, material: LinearMaterialProperties, use_b_ext: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover physical Gauss coordinates and integration volumes in mesh order."""
    gauss_coords_parts: list[np.ndarray] = []
    gauss_vols_parts: list[np.ndarray] = []
    thickness = float(material.thickness)

    for block in model.mesh.blocks:
        variant = None
        if block.ele_type.upper() == "C3D8":
            variant = "bbar" if int(use_b_ext) == 1 else "standard"
        kernel = get_kernel(block.ele_type, variant=variant)

        conn_array = np.asarray(block.connectivity, dtype=np.int32)
        if hasattr(kernel, "gauss_geometry_all"):
            coords_batch = jnp.asarray(model.mesh.nodes[conn_array], dtype=jnp.float64)
            block_coords, block_vols = jax.vmap(kernel.gauss_geometry_all)(coords_batch)
            block_coords = np.asarray(block_coords, dtype=np.float64)
            block_vols = np.asarray(block_vols, dtype=np.float64)
            if int(kernel.n_dim) == 2:
                block_vols = block_vols * thickness
            gauss_coords_parts.append(block_coords)
            gauss_vols_parts.append(block_vols)
            continue

        gp_local, gp_weights = kernel.gauss_info()
        gp_local = np.asarray(gp_local, dtype=np.float64)
        gp_weights = np.asarray(gp_weights, dtype=np.float64).reshape(-1)

        block_coords = []
        block_vols = []
        for elem_nodes in conn_array:
            node_coords = np.asarray(model.mesh.nodes[elem_nodes], dtype=np.float64)
            elem_gp_coords = []
            elem_gp_vols = []
            for gp_idx, (gp, w) in enumerate(zip(gp_local, gp_weights)):
                N = _shape_values(block.ele_type, gp)
                x_gp = N @ node_coords
                _, _, detJ = kernel.jacobian(jnp.asarray(node_coords, dtype=jnp.float64), gp_idx)
                dvol = float(detJ) * float(w)
                if int(kernel.n_dim) == 2:
                    dvol *= thickness
                elem_gp_coords.append(np.asarray(x_gp, dtype=np.float64))
                elem_gp_vols.append(dvol)
            block_coords.append(np.asarray(elem_gp_coords, dtype=np.float64))
            block_vols.append(np.asarray(elem_gp_vols, dtype=np.float64))

        gauss_coords_parts.append(np.asarray(block_coords, dtype=np.float64))
        gauss_vols_parts.append(np.asarray(block_vols, dtype=np.float64))

    if not gauss_coords_parts:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    unique_gp = {arr.shape[1] for arr in gauss_coords_parts}
    if len(unique_gp) == 1:
        gauss_coords = np.concatenate(gauss_coords_parts, axis=0)
        gauss_vols = np.concatenate(gauss_vols_parts, axis=0)
        return gauss_coords, gauss_vols

    flat_coords = [arr.reshape(-1, arr.shape[-1]) for arr in gauss_coords_parts]
    flat_vols = [arr.reshape(-1) for arr in gauss_vols_parts]
    return np.concatenate(flat_coords, axis=0), np.concatenate(flat_vols, axis=0)


def solve_linear_step(
    model: Model,
    *,
    material: LinearMaterialProperties,
    solver_options: Optional[dict] = None,
    gauss_order: int = 2,
    use_b_ext: int = 1,
    load_scale: float = 1.0,
    increment_index: int = 1,
    n_increments: int = 1,
) -> AnalysisResult:
    """Solve a single linearized increment with already-resolved material data."""
    t_start = time.time()

    family = model.metadata.get("family", "solid")
    is_shell = family == "shell"
    dimension = model.metadata.get("dimension", 3)
    ndof_per_node = 6 if is_shell else dimension
    n_nodes = model.mesh.n_nodes
    total_dofs = n_nodes * ndof_per_node

    props = material.to_dict(use_b_ext=use_b_ext, gauss_order=gauss_order)

    print(
        f"\n[{'Shell' if is_shell else 'Solid'} Linear] "
        f"increment={increment_index}/{n_increments}, load_scale={load_scale:.3f}",
        flush=True,
    )

    print("  [Assembly] Building global stiffness matrix (JAX vmap) ...", flush=True)
    t0 = time.time()
    K_bcoo = build_global_stiffness(model.mesh, props, ndof_per_node=ndof_per_node)

    data = np.asarray(K_bcoo.data)
    indices = np.asarray(K_bcoo.indices)
    K_sparse = sp.coo_matrix(
        (data, (indices[:, 0], indices[:, 1])),
        shape=(total_dofs, total_dofs),
    ).tocsr()
    print(f"  [Assembly] K built in {time.time() - t0:.2f}s, nnz={K_sparse.nnz}", flush=True)

    f_ext = build_load_vector(model, ndof_per_node) * load_scale
    print(f"  [Loads] External force vector assembled, {np.count_nonzero(f_ext)} non-zero entries.", flush=True)

    fixed_dofs, fixed_vals, free_dofs, _ = get_dof_indices(model, ndof_per_node)
    print(f"  [BCs] Fixed DOFs: {len(fixed_dofs)}, Free DOFs: {len(free_dofs)}", flush=True)

    if len(fixed_dofs) > 0 and np.any(np.abs(fixed_vals) > 1e-12):
        K_cross = K_sparse[free_dofs, :][:, fixed_dofs]
        f_ext[free_dofs] -= K_cross.dot(fixed_vals)

    K_ff = K_sparse[free_dofs, :][:, free_dofs]
    f_f = f_ext[free_dofs]

    print(f"  [Solve] Solving {len(free_dofs)} x {len(free_dofs)} system ...", flush=True)
    t0 = time.time()
    u_f = linear_solve(K_ff, f_f, solver_options)
    print(f"  [Solve] Finished in {time.time() - t0:.3f}s", flush=True)

    u_global = np.zeros(total_dofs, dtype=np.float64)
    if len(fixed_dofs) > 0:
        u_global[fixed_dofs] = fixed_vals
    u_global[free_dofs] = u_f

    f_int = K_sparse.dot(u_global)
    reaction = np.zeros(total_dofs, dtype=np.float64)
    if len(fixed_dofs) > 0:
        reaction[fixed_dofs] = f_int[fixed_dofs] - f_ext[fixed_dofs]

    print("  [Recovery] JAX vmap stress/strain recovery ...", flush=True)
    t0 = time.time()
    recover_results = recover_all_fields(
        model.mesh, jnp.array(u_global), props, ndof_per_node=ndof_per_node
    )
    t_recover = time.time() - t0

    t0 = time.time()
    gauss_strain, gauss_stress, n_gauss_per_elem, gauss_layout = _collect_gauss_fields(
        model, recover_results
    )
    gauss_coords, gauss_vols = _collect_gauss_geometry(model, material, use_b_ext)
    t_geometry = time.time() - t0
    print(f"  [Recovery] Stress/strain finished in {t_recover:.2f}s", flush=True)
    print(f"  [Recovery] Gauss geometry finished in {t_geometry:.2f}s", flush=True)
    print(f"  [Recovery] Finished in {t_recover + t_geometry:.2f}s", flush=True)

    n_str = int(gauss_stress.shape[-1]) if gauss_stress.size else 6
    c_sparse = None
    free_dofs_array = np.asarray(free_dofs, dtype=np.int32)
    if not is_shell:
        c_sparse, free_dofs_array, n_str = build_solid_c_sparse(
            model,
            ndof_per_node=ndof_per_node,
            use_b_ext=use_b_ext,
            thickness=float(material.thickness),
        )
    print(f"  [Finish] Total Time: {time.time() - t_start:.2f}s", flush=True)

    return AnalysisResult(
        u=u_global,
        reaction=reaction,
        gauss_stress=gauss_stress,
        gauss_strain=gauss_strain,
        internal_force=f_int,
        gauss_coords=gauss_coords,
        gauss_vols=gauss_vols,
        n_gauss_per_elem=n_gauss_per_elem,
        operators=Operators(
            C_sparse=c_sparse,
            K_global=K_sparse,
            free_dofs=free_dofs_array,
            n_str=n_str,
        ),
        metadata={
            "analysis_type": "linear_static",
            "family": family,
            "source_file": model.metadata.get("source_file"),
            "gauss_order": gauss_order,
            "use_b_ext": use_b_ext,
            "gauss_layout": gauss_layout,
            "n_elements": model.mesh.n_elements_total,
            "ndof_per_node": ndof_per_node,
            "increment_index": increment_index,
            "n_increments": n_increments,
            "load_scale": load_scale,
        },
    )
