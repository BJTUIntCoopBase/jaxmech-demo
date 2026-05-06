"""Visualization manifest builders for demo solid elastic MAT files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jaxmech.model.viz_manifest import build_viz_manifest, enrich_field_shapes, field_descriptor


def _has(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if value is None:
        return False
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _n_comp(payload: Mapping[str, Any], key: str, default: int) -> int:
    if not _has(payload, key):
        return int(default)
    arr = np.asarray(payload[key])
    if arr.ndim > 1 and int(arr.shape[-1]) in (1, 2, 3, 4, 6):
        return int(arr.shape[-1])
    return int(default)


def _add_if_present(fields: list[dict[str, Any]], payload: Mapping[str, Any], key: str, **kwargs) -> None:
    if _has(payload, key):
        fields.append(field_descriptor(key, **kwargs))


def build_solid_elastic_viz_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    _add_if_present(
        fields,
        payload,
        "gauss_stress",
        label="Stress",
        family="stress",
        location="gauss",
        tensor_kind="stress_voigt",
        n_components=_n_comp(payload, "gauss_stress", 6),
        default_selected=True,
        scalar_options=["components", "mises"],
    )
    _add_if_present(
        fields,
        payload,
        "gauss_strain",
        label="Strain",
        family="strain",
        location="gauss",
        tensor_kind="strain_voigt",
        n_components=_n_comp(payload, "gauss_strain", 6),
        default_selected=True,
        scalar_options=["components", "equivalent"],
    )
    _add_if_present(
        fields,
        payload,
        "solid_u_nodal",
        label="Displacement",
        family="displacement",
        location="node",
        tensor_kind="vector",
        n_components=_n_comp(payload, "solid_u_nodal", 3),
        default_selected=True,
        scalar_options=["components", "magnitude"],
    )
    _add_if_present(
        fields,
        payload,
        "solid_nforc_nodal",
        label="NFORC",
        family="force",
        location="node",
        tensor_kind="vector",
        n_components=_n_comp(payload, "solid_nforc_nodal", 3),
        default_selected=True,
        scalar_options=["components", "magnitude"],
    )
    return build_viz_manifest(
        producer_module="jaxmech.modules.inc_analysis",
        analysis_type="linear_static",
        family="solid",
        fields=enrich_field_shapes(fields, payload),
    )


def build_viz_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_solid_elastic_viz_manifest(payload)


def build_viz_manifest_from_mat(mat_path: str | Path) -> dict[str, Any]:
    import scipy.io as sio

    raw = sio.loadmat(str(Path(mat_path)), squeeze_me=False, struct_as_record=False)
    payload = {key: value for key, value in raw.items() if not key.startswith("__")}
    return build_solid_elastic_viz_manifest(payload)
