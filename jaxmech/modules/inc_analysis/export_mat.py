"""Export elastic analysis results to the standard MAT container."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix, issparse

from jaxmech.io.abaqus.inp_metadata_store import (
    delete_analysis_input_sidecar,
    load_analysis_input_sidecar_payload,
)
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult

try:
    from jaxmech.io.abaqus.shell_inp import shell_model_to_mat_dict
    _HAS_SHELL_INP = True
except ImportError:
    _HAS_SHELL_INP = False

_CPS4_LEGACY_GAUSS_ORDER = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _clean_payload(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if not k.startswith("__")}


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
    return value


def _normalize_top_level_payload(raw: dict) -> dict:
    payload = _clean_payload(raw)
    for key, value in list(payload.items()):
        payload[key] = _mat_struct_to_payload(value)
    return payload


def _load_existing_payload(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return _normalize_top_level_payload(
            sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
        )
    except Exception:
        return {}


def _payload_metadata_dict(payload: dict) -> dict[str, str]:
    metadata = _mat_struct_to_payload(payload.get("metadata"))
    if isinstance(metadata, dict):
        return {str(key): str(val) for key, val in metadata.items()}
    return {}


def _reshape_nodal_field(values: np.ndarray | None, n_nodes: int, ndof_per_node: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"AnalysisResult.{name} is required for MAT export.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != n_nodes * ndof_per_node:
            raise ValueError(f"{name} size mismatch: got {arr.size}, expected {n_nodes * ndof_per_node}.")
        return arr.reshape(n_nodes, ndof_per_node)
    if arr.ndim == 2:
        if arr.shape != (n_nodes, ndof_per_node):
            raise ValueError(f"{name} shape mismatch: got {arr.shape}, expected {(n_nodes, ndof_per_node)}.")
        return arr
    raise ValueError(f"Unsupported nodal field shape for {name}: {arr.shape}")


def _block_result_view(
    result: AnalysisResult,
    model: Model,
    block_index: int,
    field_name: str,
) -> np.ndarray:
    field = getattr(result, field_name)
    if field is None:
        raise ValueError(f"AnalysisResult.{field_name} is required for MAT export.")

    block = model.mesh.blocks[block_index]
    n_elem = block.n_elements
    elem_offset = sum(prev.n_elements for prev in model.mesh.blocks[:block_index])
    layout = str(result.metadata.get("gauss_layout", "element_gauss"))

    if layout == "element_gauss":
        return np.asarray(field[elem_offset: elem_offset + n_elem], dtype=np.float64)

    if result.n_gauss_per_elem is None:
        raise ValueError("AnalysisResult.n_gauss_per_elem is required for flat Gauss layouts.")

    block_counts = np.asarray(result.n_gauss_per_elem[elem_offset: elem_offset + n_elem], dtype=np.int32)
    unique_counts = np.unique(block_counts)
    if unique_counts.size != 1:
        raise ValueError(
            f"Mixed Gauss counts inside block {block_index} are not supported: {unique_counts.tolist()}"
        )

    n_gp = int(unique_counts[0])
    gp_offset = int(np.sum(np.asarray(result.n_gauss_per_elem[:elem_offset], dtype=np.int32)))
    block_flat = np.asarray(field[gp_offset: gp_offset + n_elem * n_gp], dtype=np.float64)
    return block_flat.reshape(n_elem, n_gp, block_flat.shape[-1])


def _block_gauss_aux_view(
    result: AnalysisResult,
    model: Model,
    block_index: int,
    field_name: str,
) -> np.ndarray | None:
    field = getattr(result, field_name)
    if field is None:
        return None

    block = model.mesh.blocks[block_index]
    n_elem = block.n_elements
    if result.n_gauss_per_elem is None:
        if str(result.metadata.get("gauss_layout", "element_gauss")) == "element_gauss":
            elem_offset = sum(prev.n_elements for prev in model.mesh.blocks[:block_index])
            return np.asarray(field[elem_offset: elem_offset + n_elem], dtype=np.float64)
        return None

    elem_offset = sum(prev.n_elements for prev in model.mesh.blocks[:block_index])
    block_counts = np.asarray(result.n_gauss_per_elem[elem_offset: elem_offset + n_elem], dtype=np.int32)
    unique_counts = np.unique(block_counts)
    if unique_counts.size != 1:
        return None
    n_gp = int(unique_counts[0])
    gp_offset = int(np.sum(np.asarray(result.n_gauss_per_elem[:elem_offset], dtype=np.int32)))
    block_flat = np.asarray(field[gp_offset: gp_offset + n_elem * n_gp], dtype=np.float64)
    return block_flat.reshape((n_elem, n_gp) + block_flat.shape[1:])


def _object_vector(items: list) -> np.ndarray:
    arr = np.empty((len(items),), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _obsolete_validation_keys() -> list[str]:
    return [
        "u",
        "solid_u_nodal",
        "solid_nforc_nodal",
        "internal_force",
        "reaction",
        "created_at_jax",
        "info_jax",
        "validated_with_ODB",
        "validated_ODBName",
        "validated_at_ODB",
        "Error_with_ODB",
        "abaqus_mat_path",
        "ele_type",
        "NodeLabels",
        "NodeCoords",
        "ElemLabels",
        "ElemConn",
        "node_map_jax_from_abaqus",
        "elem_map_jax_from_abaqus",
        "gen_u_abaqus",
        "gen_u_jax",
        "gen_u_diff",
        "gen_strain_abaqus",
        "gen_strain_jax",
        "gen_strain_diff",
        "gen_stress_abaqus",
        "gen_stress_jax",
        "gen_stress_diff",
        "gen_internal_force_abaqus",
        "gen_internal_force_jax",
        "gen_internal_force_diff",
        "layer_strain_abaqus",
        "layer_strain_jax",
        "layer_strain_diff",
        "layer_stress_abaqus",
        "layer_stress_jax",
        "layer_stress_diff",
        "tri3_gp_align",
        "tri3_basis_angles_deg",
        "NodalDisp_jax",
        "NFORC_jax",
        "mixed_element_jax",
        "cell_block_cell_types_jax",
        "cell_block_ele_types_jax",
        "cell_block_cells_jax",
        "cell_type_names_jax",
        "cell_type_cells_jax",
        "EleNodeDispSetByType_jax",
        "EleGaussStressSetByType_jax",
        "EleGaussStrainSetByType_jax",
        "n_gauss_per_elem_by_type_jax",
        "EleNodeDispSet_jax",
        "GenStrainSet_jax",
        "GenStressSet_jax",
        "EleGaussStrainSet_jax",
        "EleGaussStressSet_jax",
        "gauss_coords_jax",
        "gauss_coords_by_type_jax",
        "gauss_vols_jax",
        "gauss_vols_by_type_jax",
        "n_gauss_per_elem_jax",
        "free_dofs_jax",
        "C_sparse_jax",
        "n_str_jax",
        "C_sparse",
        "C_sparse_shape",
        "C_sparse_row",
        "C_sparse_col",
        "C_sparse_data",
        "G_int",
        "G_int_shape",
        "G_int_row",
        "G_int_col",
        "G_int_data",
        "abaqus_u_nodal",
        "abaqus_nforc",
        "abaqus_nforcso",
        "abaqus_layer_stress",
        "abaqus_layer_strain",
        "abaqus_gen_section_force",
        "abaqus_gen_section_moment",
        "abaqus_gen_strain",
        "abaqus_gen_curvature",
        "abaqus_gauss_stress",
        "abaqus_gauss_strain",
        "abaqus_section_z_normalized",
        "abaqus_node_labels",
        "abaqus_node_coords",
        "abaqus_elem_labels",
        "abaqus_elem_conn",
        "abaqus_elem_types",
        "abaqus_frame_index",
        "abaqus_frame_time",
        "abaqus_frame_value",
        "abaqus_instance_name",
        "abaqus_step_name",
        "abaqus_ip_order",
        "abaqus_voigt_order",
        "abaqus_voigt_perm",
        "abaqus_jax_from_aba",
        "abaqus_n_gp",
        "abaqus_n_sp",
        "abaqus_n_nodes",
        "abaqus_n_elem",
        "info_abaqus",
        "abaqus_odb_path",
        "field_output_keys_abaqus",
    ]


def _simplify_inp_data(raw: dict) -> dict:
    simplified: dict[str, object] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = value
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 2 and 1 in arr.shape:
            simplified[key] = arr.reshape(-1)
        else:
            simplified[key] = arr
    return simplified


def _legacy_gauss_order(ele_type: str, field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim < 3:
        return arr
    if str(ele_type).upper() in {"CPS4", "CPE4"} and arr.shape[1] == 4:
        return arr[:, _CPS4_LEGACY_GAUSS_ORDER, ...]
    return arr


def export_solid_elastic_result_mat(
    model: Model,
    result: AnalysisResult,
    out_mat_path: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> Path:
    """Write one solid elastic MAT container with InpData and JAX result fields."""
    if str(model.metadata.get("family", "solid")) != "solid":
        raise NotImplementedError("export_solid_elastic_result_mat currently supports solid models only.")

    out_path = Path(out_mat_path)
    payload = _load_existing_payload(out_path)
    sidecar_payload = load_analysis_input_sidecar_payload(out_path)
    for key, value in sidecar_payload.items():
        if key == "metadata":
            continue
        payload[key] = value
    sidecar_meta = _mat_struct_to_payload(sidecar_payload.get("metadata"))
    if isinstance(sidecar_meta, dict):
        payload["metadata"] = {
            **_payload_metadata_dict(payload),
            **{str(key): str(val) for key, val in sidecar_meta.items()},
        }

    for key in _obsolete_validation_keys():
        payload.pop(key, None)

    raw_inp = model.metadata.get("_solid_inp_data")
    if isinstance(raw_inp, dict):
        payload["InpData"] = _simplify_inp_data(raw_inp)

    ndof_per_node = int(result.metadata.get("ndof_per_node", model.mesh.n_dim))
    n_nodes = model.mesh.n_nodes
    nodal_disp = _reshape_nodal_field(result.u, n_nodes, ndof_per_node, "u")
    nodal_force = -_reshape_nodal_field(result.internal_force, n_nodes, ndof_per_node, "internal_force")

    generic_meta = {
        **_payload_metadata_dict(payload),
        "source_file": str(model.metadata.get("source_file", "")),
        "family": "solid",
        "analysis_type": str(result.metadata.get("analysis_type", "linear_static")),
        "gauss_order": str(result.metadata.get("gauss_order", "")),
        "use_b_ext": str(result.metadata.get("use_b_ext", "")),
        "analysis_stage": "result",
        "has_analysis_result": "1",
        "analysis_input_sidecar": "0",
    }
    if extra_metadata:
        generic_meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    solid_elem_types = np.asarray(
        [str(block.ele_type).upper() for block in model.mesh.blocks for _ in range(block.n_elements)],
        dtype=object,
    )
    payload.update(
        {
            "metadata": {key: str(val) for key, val in generic_meta.items()},
            "gauss_stress": np.asarray(result.gauss_stress) if result.gauss_stress is not None else None,
            "gauss_strain": np.asarray(result.gauss_strain) if result.gauss_strain is not None else None,
            "gauss_coords": np.asarray(result.gauss_coords) if result.gauss_coords is not None else None,
            "gauss_vols": np.asarray(result.gauss_vols) if result.gauss_vols is not None else None,
        "n_gauss_per_elem": (
            np.asarray(result.n_gauss_per_elem) if result.n_gauss_per_elem is not None else None
        ),
        "gauss_coords": np.asarray(result.gauss_coords) if result.gauss_coords is not None else None,
        "gauss_vols": np.asarray(result.gauss_vols) if result.gauss_vols is not None else None,
        "solid_u_nodal": nodal_disp,
        "solid_nforc_nodal": nodal_force,
        "solid_elem_types": solid_elem_types,
        "free_dofs": np.asarray(result.operators.free_dofs, dtype=np.int64).reshape(1, -1)
        if result.operators.free_dofs is not None else None,
        "C_sparse": result.operators.C_sparse.tocsr() if result.operators.C_sparse is not None else None,
    }
    )

    from jaxmech.model.viz_manifest import attach_viz_manifest
    from jaxmech.modules.inc_analysis.visualize_mat import build_solid_elastic_viz_manifest

    attach_viz_manifest(payload, build_solid_elastic_viz_manifest(payload))
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), clean_payload, do_compression=True)
    delete_analysis_input_sidecar(out_path)
    return out_path


def export_shell_elastic_result_mat(
    model: Model,
    result: AnalysisResult,
    out_mat_path: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> Path:
    """Write one shell elastic MAT container with InpData and JAX result fields."""
    if str(model.metadata.get("family", "")).lower() != "shell":
        raise NotImplementedError("export_shell_elastic_result_mat currently supports shell models only.")

    shell_data = model.metadata.get("_shell_inp_data")
    if shell_data is None:
        raise ValueError("Shell MAT export requires Model.metadata['_shell_inp_data'].")
    if result.shell is None:
        raise ValueError("Shell MAT export requires AnalysisResult.shell recovery fields.")

    out_path = Path(out_mat_path)
    payload = _load_existing_payload(out_path)
    sidecar_payload = load_analysis_input_sidecar_payload(out_path)
    for key, value in sidecar_payload.items():
        if key == "metadata":
            continue
        payload[key] = value
    sidecar_meta = _mat_struct_to_payload(sidecar_payload.get("metadata"))
    if isinstance(sidecar_meta, dict):
        payload["metadata"] = {
            **_payload_metadata_dict(payload),
            **{str(key): str(val) for key, val in sidecar_meta.items()},
        }

    for key in _obsolete_validation_keys():
        payload.pop(key, None)

    shell = result.shell
    payload["InpData"] = shell_model_to_mat_dict(shell_data)

    generic_meta = {
        **_payload_metadata_dict(payload),
        "source_file": str(model.metadata.get("source_file", "")),
        "family": "shell",
        "analysis_type": str(result.metadata.get("analysis_type", "linear_static")),
        "analysis_stage": "result",
        "has_analysis_result": "1",
        "analysis_input_sidecar": "0",
    }
    if extra_metadata:
        generic_meta.update({str(k): str(v) for k, v in extra_metadata.items()})

    payload.update(
        {
            "metadata": {key: str(val) for key, val in generic_meta.items()},
            "gauss_stress": np.asarray(result.gauss_stress) if result.gauss_stress is not None else None,
            "gauss_strain": np.asarray(result.gauss_strain) if result.gauss_strain is not None else None,
            "internal_force": np.asarray(result.internal_force) if result.internal_force is not None else None,
            "reaction": np.asarray(result.reaction) if result.reaction is not None else None,
            "gauss_coords": np.asarray(result.gauss_coords) if result.gauss_coords is not None else None,
            "gauss_vols": np.asarray(result.gauss_vols) if result.gauss_vols is not None else None,
            "n_gauss_per_elem": (
                np.asarray(result.n_gauss_per_elem) if result.n_gauss_per_elem is not None else None
            ),
            "shell_u_nodal": np.asarray(shell.u_nodal) if shell.u_nodal is not None else None,
            "shell_gen_internal_force": (
                np.asarray(shell.gen_internal_force) if shell.gen_internal_force is not None else None
            ),
            "shell_layer_strain": np.asarray(shell.layer_strain) if shell.layer_strain is not None else None,
            "shell_layer_stress": np.asarray(shell.layer_stress) if shell.layer_stress is not None else None,
            "shell_section_z": np.asarray(shell.section_z) if shell.section_z is not None else None,
            "shell_elem_thickness": (
                np.asarray(shell.elem_thickness) if shell.elem_thickness is not None else None
            ),
            "C_gen": result.operators.C_gen.tocsr() if result.operators.C_gen is not None else None,
        }
    )

    from jaxmech.model.viz_manifest import attach_viz_manifest
    from jaxmech.modules.inc_analysis.visualize_mat import build_shell_elastic_viz_manifest

    attach_viz_manifest(payload, build_shell_elastic_viz_manifest(payload))
    clean_payload = {k: v for k, v in payload.items() if v is not None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), clean_payload, do_compression=True)
    delete_analysis_input_sidecar(out_path)
    return out_path
