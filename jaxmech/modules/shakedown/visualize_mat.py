"""Visualization manifest builders for demo solid shakedown MAT files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jaxmech.model.viz_manifest import build_viz_manifest, field_descriptor


def _lookup_in_obj(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        values = [_lookup_in_obj(item, key) for item in obj.flat]
        values = [item for item in values if item is not None]
        return values[0] if values else None
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None) and key in obj.dtype.names:
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    return None


def _lookup(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    for parent in ("ElasticInputSet", "ResultsSet", "ConfigInfo"):
        value = _lookup_in_obj(payload.get(parent), key)
        if value is not None:
            return value
    return None


def _has(payload: Mapping[str, Any], key: str) -> bool:
    value = _lookup(payload, key)
    if value is None:
        return False
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _n_str(payload: Mapping[str, Any]) -> int:
    for key in ("residual_stress", "sigma_E", "gauss_stress_cases"):
        value = _lookup(payload, key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim >= 2 and int(arr.shape[-1]) in (3, 6):
            return int(arr.shape[-1])
        if arr.ndim >= 2 and int(arr.shape[1]) in (3, 6):
            return int(arr.shape[1])
    return 6


def build_shakedown_viz_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Declare visual fields for solid CVXPY shakedown summaries."""
    n_comp = _n_str(payload)
    fields: list[dict[str, Any]] = []
    if _has(payload, "residual_stress"):
        fields.append(
            field_descriptor(
                "shakedown_residual_stress",
                label="Shakedown residual stress",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                frames="single",
                default_selected=True,
                source_key="residual_stress",
                scalar_options=["components", "mises"],
                symbol=r"\rho",
                formula=r"\rho=\sigma^{tot}-\alpha\sigma^E",
                description="Self-equilibrated residual stress from the lower-bound shakedown solve",
            )
        )
    if _has(payload, "residual_stress") and _has(payload, "gauss_stress_cases"):
        fields.append(
            field_descriptor(
                "shakedown_total_stress",
                label="Shakedown total stress",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_keys=["residual_stress", "gauss_stress_cases", "objective_value", "load_factor"],
                scalar_options=["components", "mises"],
                symbol=r"\sigma^{tot}",
                formula=r"\sigma^{tot}_v=\rho+\alpha\sigma^E_v",
                description="Total stress reconstructed at each load-domain vertex",
            )
        )
    if _has(payload, "yield_ratio"):
        fields.append(
            field_descriptor(
                "shakedown_inequality_violation",
                label="Yield inequality violation",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_key="yield_ratio",
                scalar_options=["scalar"],
                symbol=r"C_{ineq}",
                formula=r"C_{ineq}=\max(\sigma_{eq}/\sigma_y-1,0)",
                description="Zero means the Gauss point satisfies the yield constraint",
            )
        )
    if _has(payload, "sigma_E"):
        fields.append(
            field_descriptor(
                "sigma_E",
                label="Elastic stress for shakedown input",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                default_selected=False,
                scalar_options=["components", "mises"],
            )
        )
    return build_viz_manifest(
        producer_module="jaxmech.modules.shakedown",
        analysis_type="shakedown",
        family="solid",
        fields=fields,
        frame_axes={"vertex": {"label": "Vertex"}},
        metadata={"family": "solid"},
    )


def build_viz_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_shakedown_viz_manifest(payload)


def build_viz_manifest_from_mat(mat_path: str | Path) -> dict[str, Any]:
    import scipy.io as sio

    raw = sio.loadmat(str(Path(mat_path)), squeeze_me=False, struct_as_record=True)
    payload = {key: value for key, value in raw.items() if not key.startswith("__")}
    return build_shakedown_viz_manifest(payload)
