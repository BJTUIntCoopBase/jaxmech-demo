"""Export demo solid elastic analysis results to MAT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import issparse

from jaxmech.io.abaqus.inp_metadata_store import (
    delete_analysis_input_sidecar,
    load_analysis_input_sidecar_payload,
)
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult


def _clean_payload(raw: dict) -> dict:
    return {key: value for key, value in raw.items() if not key.startswith("__")}


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
        return {key: val for key, val in out.__dict__.items() if not key.startswith("_")}
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
        return _normalize_top_level_payload(sio.loadmat(str(path), squeeze_me=False, struct_as_record=False))
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
    if arr.ndim == 2 and arr.shape == (n_nodes, ndof_per_node):
        return arr
    raise ValueError(f"Unsupported nodal field shape for {name}: {arr.shape}")


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


def export_solid_elastic_result_mat(
    model: Model,
    result: AnalysisResult,
    out_mat_path: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> Path:
    """Write one solid elastic MAT with InpData, result fields, and viz manifest."""
    if str(model.metadata.get("family", "solid")).lower() != "solid":
        raise NotImplementedError("The demo exporter supports solid models only.")

    out_path = Path(out_mat_path)
    payload = _load_existing_payload(out_path)
    sidecar_payload = load_analysis_input_sidecar_payload(out_path)
    for key, value in sidecar_payload.items():
        if key != "metadata":
            payload[key] = value
    sidecar_meta = _mat_struct_to_payload(sidecar_payload.get("metadata"))
    if isinstance(sidecar_meta, dict):
        payload["metadata"] = {
            **_payload_metadata_dict(payload),
            **{str(key): str(val) for key, val in sidecar_meta.items()},
        }

    raw_inp = model.metadata.get("_solid_inp_data")
    if isinstance(raw_inp, dict):
        payload["InpData"] = _simplify_inp_data(raw_inp)

    ndof_per_node = int(result.metadata.get("ndof_per_node", model.mesh.n_dim))
    n_nodes = model.mesh.n_nodes
    nodal_disp = _reshape_nodal_field(result.u, n_nodes, ndof_per_node, "u")
    nodal_force = -_reshape_nodal_field(result.internal_force, n_nodes, ndof_per_node, "internal_force")

    metadata = {
        **_payload_metadata_dict(payload),
        "source_file": str(model.metadata.get("source_file", "")),
        "family": "solid",
        "analysis_type": "linear_static",
        "gauss_order": str(result.metadata.get("gauss_order", "")),
        "use_b_ext": str(result.metadata.get("use_b_ext", "")),
        "analysis_stage": "result",
        "has_analysis_result": "1",
        "analysis_input_sidecar": "0",
    }
    if extra_metadata:
        metadata.update({str(key): str(val) for key, val in extra_metadata.items()})

    solid_elem_types = np.asarray(
        [str(block.ele_type).upper() for block in model.mesh.blocks for _ in range(block.n_elements)],
        dtype=object,
    )
    payload.update(
        {
            "metadata": {key: str(val) for key, val in metadata.items()},
            "gauss_stress": np.asarray(result.gauss_stress) if result.gauss_stress is not None else None,
            "gauss_strain": np.asarray(result.gauss_strain) if result.gauss_strain is not None else None,
            "gauss_coords": np.asarray(result.gauss_coords) if result.gauss_coords is not None else None,
            "gauss_vols": np.asarray(result.gauss_vols) if result.gauss_vols is not None else None,
            "n_gauss_per_elem": np.asarray(result.n_gauss_per_elem) if result.n_gauss_per_elem is not None else None,
            "solid_u_nodal": nodal_disp,
            "solid_nforc_nodal": nodal_force,
            "solid_elem_types": solid_elem_types,
            "free_dofs": np.asarray(result.operators.free_dofs, dtype=np.int64).reshape(1, -1)
            if result.operators.free_dofs is not None
            else None,
            "C_sparse": result.operators.C_sparse.tocsr() if result.operators.C_sparse is not None else None,
        }
    )

    from jaxmech.model.viz_manifest import attach_viz_manifest
    from jaxmech.modules.inc_analysis.visualize_mat import build_solid_elastic_viz_manifest

    attach_viz_manifest(payload, build_solid_elastic_viz_manifest(payload))
    clean_payload = {key: value for key, value in payload.items() if value is not None}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), clean_payload, do_compression=True)
    delete_analysis_input_sidecar(out_path)
    return out_path


__all__ = ["export_solid_elastic_result_mat"]
