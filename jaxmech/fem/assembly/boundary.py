"""
jaxmech.fem.assembly.boundary - Boundary condition and load processing.
"""

from __future__ import annotations

import numpy as np

from jaxmech.fem.assembly.surface_loads import assemble_surface_load_mixed
from jaxmech.model.model import Model


def _resolve_nodes(mesh, node_set) -> np.ndarray:
    """Resolve a node-set name or ABAQUS node label to 0-based node indices."""
    if isinstance(node_set, str):
        set_name = node_set.upper()
        if set_name in mesh.node_sets:
            return np.asarray(mesh.node_sets[set_name], dtype=np.int32).reshape(-1)
        try:
            node_label = int(set_name)
        except ValueError as exc:
            raise ValueError(f"Unknown node set or node label: {set_name}") from exc
        if mesh.node_ids is None:
            raise ValueError(f"Model does not expose node_ids; cannot resolve node label {node_label}.")
        node_ids = np.asarray(mesh.node_ids).reshape(-1)
        matches = np.where(node_ids == node_label)[0]
        if matches.size == 0:
            raise ValueError(f"Unknown node label referenced by load/BC: {node_label}")
        return matches.astype(np.int32, copy=False)
    return np.asarray(node_set, dtype=np.int32).reshape(-1)


def get_dof_indices(model: Model, ndof_per_node: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract fixed and free DOFs based on Model BCs.

    Returns:
        fixed_dofs: (N_fixed,) integer array of global DOFs that are fixed.
        fixed_values: (N_fixed,) float array of prescribed values for those DOFs.
        free_dofs: (N_free,) integer array of global DOFs that are free.
        dof_to_free: (N_total,) integer mapping global DOF -> free DOF index (-1 if fixed).
    """
    n_nodes = model.mesh.n_nodes
    total_dofs = n_nodes * ndof_per_node

    fixed_dofs_list = []
    fixed_values_list = []

    for bc in model.bcs:
        nodes = np.atleast_1d(_resolve_nodes(model.mesh, bc.node_set))

        for dof_offset in bc.dof_range_0based:
            if dof_offset >= ndof_per_node:
                continue
            global_dofs = nodes * ndof_per_node + dof_offset
            fixed_dofs_list.extend(global_dofs.tolist())
            fixed_values_list.extend([bc.value] * len(global_dofs))

    if not fixed_dofs_list:
        fixed_dofs = np.array([], dtype=np.int32)
        fixed_values = np.array([], dtype=np.float64)
    else:
        fixed_dofs_arr = np.array(fixed_dofs_list, dtype=np.int32)
        fixed_values_arr = np.array(fixed_values_list, dtype=np.float64)
        _, unique_idx = np.unique(fixed_dofs_arr, return_index=True)
        fixed_dofs = fixed_dofs_arr[unique_idx]
        fixed_values = fixed_values_arr[unique_idx]

    all_dofs = np.arange(total_dofs, dtype=np.int32)
    free_mask = np.ones(total_dofs, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = all_dofs[free_mask]

    dof_to_free = np.full(total_dofs, -1, dtype=np.int32)
    dof_to_free[free_dofs] = np.arange(len(free_dofs), dtype=np.int32)

    return fixed_dofs, fixed_values, free_dofs, dof_to_free


def build_load_vector(model: Model, ndof_per_node: int) -> np.ndarray:
    """
    Build the global external force vector.
    Supports concentrated loads and parsed surface loads.
    """
    n_nodes = model.mesh.n_nodes
    total_dofs = n_nodes * ndof_per_node
    f_ext = np.zeros(total_dofs, dtype=np.float64)

    if not model.load_cases:
        return f_ext

    load_case = model.load_cases[0]

    for cl in load_case.concentrated:
        nodes = np.atleast_1d(_resolve_nodes(model.mesh, cl.node_set))

        dofs = [cl.dof] if isinstance(cl.dof, int) else cl.dof
        for dof_idx in dofs:
            if dof_idx >= ndof_per_node:
                continue
            global_dofs = nodes * ndof_per_node + dof_idx
            mags = cl.magnitude
            if np.isscalar(mags) or (isinstance(mags, np.ndarray) and mags.size == 1):
                f_ext[global_dofs] += float(mags)
            else:
                f_ext[global_dofs] += np.asarray(mags).flatten()

    if load_case.surface_loads:
        _add_surface_loads(f_ext, model, load_case, ndof_per_node)

    return f_ext


def _add_surface_loads(f_ext: np.ndarray, model: Model, load_case, ndof_per_node: int):
    """
    Assemble surface loads from parsed input data.

    The solid path must consume the parsed INP payload already attached to the
    model. Downstream steps are not allowed to reopen the source ``.inp``.
    """
    family = model.metadata.get("family", "solid")
    if family == "shell":
        _add_shell_surface_loads(f_ext, model, load_case, ndof_per_node)
        return

    raw = model.metadata.get("_solid_inp_data")
    if raw is None:
        raw = model.mesh.metadata.get("_solid_inp_data")
    if raw is None:
        print("Warning: Surface loads ignored because parsed solid input data is missing in metadata.")
        return

    try:
        dsload_entries = [
            (sl.surface_name, sl.load_type, sl.magnitude)
            for sl in load_case.surface_loads
        ]
        f_surf = assemble_surface_load_mixed(
            points=model.mesh.nodes,
            blocks=model.mesh.blocks,
            parsed_data=raw,
            dsload_entries=dsload_entries,
        )

        ndim = model.mesh.nodes.shape[1]
        f_surf_2d = f_surf.reshape(-1, ndim)
        for i in range(min(ndim, ndof_per_node)):
            f_ext[i::ndof_per_node] += f_surf_2d[:, i]

    except Exception as e:
        print(f"Warning: Failed to process surface loads from parsed data: {e}")
        import traceback
        traceback.print_exc()


def _add_shell_surface_loads(f_ext: np.ndarray, model: Model, load_case, ndof_per_node: int) -> None:
    """Assemble shell pressure loads from parsed shell metadata."""
    shell_data = model.metadata.get("_shell_inp_data")
    if shell_data is None:
        print("Warning: Shell surface loads ignored because parsed shell input data is missing in metadata.")
        return

    if not getattr(shell_data, "_elem_label_to_idx", None):
        shell_data.build_index_maps()

    n_total = model.mesh.n_nodes * ndof_per_node
    if f_ext.shape[0] != n_total:
        raise ValueError(f"Shell load vector size mismatch: got {f_ext.shape[0]}, expected {n_total}.")

    for surface_load in load_case.surface_loads:
        surf_key = str(surface_load.surface_name).upper()
        if surf_key not in shell_data.surfaces:
            continue

        side, elem_labels_surf = shell_data.surfaces[surf_key]
        load_type = str(surface_load.load_type).upper()

        for elem_label in np.asarray(elem_labels_surf).reshape(-1):
            elem_idx = shell_data._elem_label_to_idx.get(int(elem_label))
            if elem_idx is None:
                continue
            node_indices = np.asarray(shell_data.elem_conn[elem_idx], dtype=np.int32)
            coords = np.asarray(model.mesh.nodes[node_indices], dtype=np.float64)
            n_nodes_elem = len(node_indices)

            if n_nodes_elem == 3:
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                normal = np.cross(v1, v2)
                normal_mag = np.linalg.norm(normal)
                if normal_mag > 1e-30:
                    normal_hat = normal / normal_mag
                else:
                    normal_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                area = 0.5 * normal_mag
                n_node_div = 3.0
            elif n_nodes_elem == 4:
                v1 = coords[2] - coords[0]
                v2 = coords[3] - coords[1]
                normal = np.cross(v1, v2)
                normal_mag = np.linalg.norm(normal)
                if normal_mag > 1e-30:
                    normal_hat = normal / normal_mag
                else:
                    normal_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                area = 0.5 * normal_mag
                n_node_div = 4.0
            else:
                raise NotImplementedError(f"Unsupported shell element with {n_nodes_elem} nodes for pressure loads.")

            if load_type == "P":
                sign = 1.0 if str(side).upper() == "SNEG" else -1.0
                traction_vec = sign * float(surface_load.magnitude) * normal_hat
            elif load_type == "TRVEC":
                if surface_load.direction is None:
                    continue
                direction = np.asarray(surface_load.direction, dtype=np.float64).reshape(-1)
                if direction.size < 3:
                    continue
                dir_norm = np.linalg.norm(direction[:3])
                if dir_norm <= 1.0e-30:
                    continue
                traction_vec = float(surface_load.magnitude) * (direction[:3] / dir_norm)
            else:
                continue

            force_per_node = traction_vec * area / n_node_div

            for node_idx in node_indices:
                for comp in range(min(3, ndof_per_node)):
                    f_ext[int(node_idx) * ndof_per_node + comp] += force_per_node[comp]
