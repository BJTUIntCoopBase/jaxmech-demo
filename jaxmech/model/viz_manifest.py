"""Lightweight visualization manifest helpers for MAT files.

The manifest is intentionally independent from the PyVista-based visualization
runtime.  Solver/export modules use it to declare which MAT variables are part
of the public visualization contract; the web visualization reader consumes the
same declaration before falling back to legacy heuristics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

VIZ_MANIFEST_KEY = "VizManifestJSON"
VIZ_MANIFEST_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def encode_viz_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a visualization manifest to compact JSON."""
    return json.dumps(
        dict(manifest),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _coerce_text(value: Any) -> str:
    cur = value
    for _ in range(8):
        if isinstance(cur, np.ndarray):
            if cur.dtype.kind in {"U", "S"}:
                if cur.ndim == 0:
                    cur = cur.item()
                elif cur.size == 1:
                    cur = cur.reshape(-1)[0]
                else:
                    return "".join(str(x) for x in cur.reshape(-1)).strip()
                continue
            if cur.dtype == object and cur.size == 1:
                cur = cur.reshape(-1)[0]
                continue
        break
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore").strip()
    return str(cur).strip() if cur is not None else ""


def decode_viz_manifest(value: Any) -> dict[str, Any] | None:
    """Decode a MAT-loaded ``VizManifestJSON`` value."""
    text = _coerce_text(value)
    if not text:
        return None
    try:
        decoded = json.loads(text)
    except Exception:
        return None
    if not isinstance(decoded, dict):
        return None
    fields = decoded.get("fields")
    if not isinstance(fields, list):
        return None
    return decoded


def extract_viz_manifest(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the manifest from a loaded MAT dict, if present and valid."""
    if VIZ_MANIFEST_KEY not in raw:
        return None
    return decode_viz_manifest(raw.get(VIZ_MANIFEST_KEY))


def attach_viz_manifest(payload: dict[str, Any], manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Attach *manifest* to *payload* as ``VizManifestJSON`` and return payload."""
    if manifest is None:
        payload.pop(VIZ_MANIFEST_KEY, None)
    else:
        payload[VIZ_MANIFEST_KEY] = encode_viz_manifest(manifest)
    return payload


def _shape_of(payload: Mapping[str, Any], key: str | None) -> list[int] | None:
    if not key or key not in payload:
        return None
    try:
        arr = np.asarray(payload[key])
    except Exception:
        return None
    if arr.size == 0:
        return None
    return [int(v) for v in arr.shape]


def field_descriptor(
    key: str,
    *,
    label: str,
    family: str,
    location: str,
    tensor_kind: str,
    n_components: int | None = None,
    frames: str = "single",
    frame_axis: str | None = None,
    default_selected: bool = False,
    source_key: str | None = None,
    source_keys: Sequence[str] | None = None,
    shape: Sequence[int] | None = None,
    scalar_options: Sequence[str] | None = None,
    symbol: str | None = None,
    formula: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build one field entry for a visualization manifest."""
    entry: dict[str, Any] = {
        "key": str(key),
        "label": str(label),
        "family": str(family),
        "location": str(location),
        "tensor_kind": str(tensor_kind),
        "frames": str(frames),
        "default_selected": bool(default_selected),
    }
    if n_components is not None:
        entry["n_components"] = int(n_components)
    if frame_axis:
        entry["frame_axis"] = str(frame_axis)
    if source_key:
        entry["source_key"] = str(source_key)
    if source_keys:
        entry["source_keys"] = [str(item) for item in source_keys if str(item).strip()]
    if shape is not None:
        entry["shape"] = [int(v) for v in shape]
    if scalar_options:
        entry["scalar_options"] = [str(item) for item in scalar_options]
    if symbol:
        entry["symbol"] = str(symbol)
    if formula:
        entry["formula"] = str(formula)
    if description:
        entry["description"] = str(description)
    return entry


def build_viz_manifest(
    *,
    producer_module: str,
    analysis_type: str,
    family: str = "",
    fields: Sequence[Mapping[str, Any]],
    frame_axes: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest dict."""
    manifest: dict[str, Any] = {
        "schema_version": VIZ_MANIFEST_SCHEMA_VERSION,
        "producer_module": str(producer_module),
        "analysis_type": str(analysis_type),
        "family": str(family),
        "fields": [dict(field) for field in fields],
    }
    if frame_axes:
        manifest["frame_axes"] = dict(frame_axes)
    if metadata:
        manifest["metadata"] = {str(k): _json_default(v) for k, v in metadata.items()}
    return manifest


def enrich_field_shapes(fields: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fill missing ``shape`` values from payload arrays when possible."""
    out: list[dict[str, Any]] = []
    for field in fields:
        entry = dict(field)
        if "shape" not in entry:
            shape = _shape_of(payload, entry.get("source_key") or entry.get("key"))
            if shape is not None:
                entry["shape"] = shape
        out.append(entry)
    return out


def load_viz_manifest_from_mat(mat_path: str | Path) -> dict[str, Any] | None:
    """Load only ``VizManifestJSON`` from *mat_path*."""
    import scipy.io as sio

    raw = sio.loadmat(
        str(mat_path),
        squeeze_me=False,
        struct_as_record=True,
        variable_names=[VIZ_MANIFEST_KEY],
    )
    return extract_viz_manifest(raw)


def write_viz_manifest_to_mat(mat_path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Persist ``VizManifestJSON`` into an existing MAT file."""
    import scipy.io as sio

    mat_path = Path(mat_path)
    raw = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    payload = {key: value for key, value in raw.items() if not key.startswith("__")}
    attach_viz_manifest(payload, manifest)
    sio.savemat(str(mat_path), payload, do_compression=True)
    return mat_path
