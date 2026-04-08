"""Persist lightweight INP metadata and editable analysis inputs in MAT files.

This module keeps the Windows-side INP parse step independent from solver-heavy
imports. The parse result is stored as a sidecar MAT under inc_analysis/inputs/,
and exporters later merge the same input snapshot into the final analysis MAT.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from scipy.sparse import issparse

_ANALYSIS_INPUT_JSON_KEY = "AnalysisInputJSON"
_ANALYSIS_INPUT_HISTORY_JSON_KEY = "AnalysisInputHistoryJSON"
_INP_METADATA_JSON_KEY = "InpMetadataJSON"

_TRACKED_INPUT_KEYS = (
    "material_model",
    "E_override",
    "nu_override",
    "yield_stress_override",
    "use_b_ext",
    "gauss_order",
    "n_increments",
    "max_iterations",
    "convergence_tol",
)

_RESULT_FIELD_KEYS = (
    "C_sparse",
    "C_gen",
    "PlasticResult",
    "solid_u_nodal",
    "shell_u_nodal",
    "gauss_stress",
)


def analysis_input_sidecar_path(result_mat_path: str | Path) -> Path:
    """Return the sidecar MAT path used by the parse pre-step."""
    result_path = Path(result_mat_path)
    return result_path.parent / "inputs" / result_path.name


def delete_analysis_input_sidecar(result_mat_path: str | Path) -> None:
    """Delete the temporary parse sidecar after a successful export."""
    sidecar_path = analysis_input_sidecar_path(result_mat_path)
    if sidecar_path.is_file():
        sidecar_path.unlink()


def _clean_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def _unwrap_scalar(value: Any) -> Any:
    cur = value
    while isinstance(cur, np.ndarray) and cur.size == 1:
        cur = cur.reshape(-1)[0]
    return cur


def _mat_value_to_payload(value: Any) -> Any:
    out = _unwrap_scalar(value)
    if issparse(out):
        return out
    if isinstance(out, dict):
        return out
    if hasattr(out, "__dict__"):
        return {key: val for key, val in out.__dict__.items() if not key.startswith("_")}
    return value


def _normalize_top_level_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = _clean_payload(raw)
    for key, value in list(payload.items()):
        payload[key] = _mat_value_to_payload(value)
    return payload


def _struct_to_dict(value: Any) -> dict[str, Any]:
    cur = _unwrap_scalar(value)
    if isinstance(cur, dict):
        return {str(key): _unwrap_scalar(val) for key, val in cur.items()}
    if hasattr(cur, "_fieldnames"):
        return {str(name): _unwrap_scalar(getattr(cur, name)) for name in getattr(cur, "_fieldnames", [])}
    if isinstance(cur, np.ndarray) and getattr(cur.dtype, "names", None):
        rec = cur.reshape(-1)[0]
        return {str(name): _unwrap_scalar(rec[name]) for name in cur.dtype.names or []}
    return {}


def _string_value(value: Any) -> str:
    cur = _unwrap_scalar(value)
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore").strip()
    if isinstance(cur, str):
        return cur.strip()
    if isinstance(cur, np.ndarray):
        if cur.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in cur.reshape(-1)).strip()
        if cur.dtype == object and cur.size == 1:
            return _string_value(cur.reshape(-1)[0])
    return str(cur).strip()


def _json_field(value: Any) -> np.ndarray:
    cell = np.empty((1,), dtype=object)
    cell[0] = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return cell


def _parse_json_field(payload: dict[str, Any], key: str) -> Any:
    raw = payload.get(key)
    if raw is None:
        return None
    text = _string_value(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan"}:
            return None
        number = float(text)
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def _has_explicit_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "nan"}


def _positive_int(value: Any, *, default: int) -> int:
    try:
        number = int(float(str(value).strip()))
    except Exception:
        return int(default)
    return number if number > 0 else int(default)


def _positive_float(value: Any, *, default: float) -> float:
    try:
        number = float(str(value).strip())
    except Exception:
        return float(default)
    return number if number > 0.0 else float(default)


def _binary_flag(value: Any, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return 1
    if text in {"0", "false", "no", "off", ""}:
        return 0
    try:
        return 1 if int(float(text)) != 0 else 0
    except Exception:
        return int(default)


def _optional_positive_int(value: Any) -> int | None:
    if not _has_explicit_value(value):
        return None
    try:
        number = int(float(str(value).strip()))
    except Exception:
        return None
    return number if number > 0 else None


def _optional_positive_float(value: Any) -> float | None:
    if not _has_explicit_value(value):
        return None
    try:
        number = float(str(value).strip())
    except Exception:
        return None
    return number if number > 0.0 else None


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _payload_has_analysis_result(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in _RESULT_FIELD_KEYS)


def _build_analysis_input(inp_meta: dict[str, Any], analysis_input: dict[str, Any], *, source: str) -> dict[str, Any]:
    materials = inp_meta.get("materials") or []
    material = materials[0] if materials else {}
    elastic = material.get("elastic") or {}
    plastic = material.get("plastic") or {}
    step = inp_meta.get("step") or {}

    material_model = str(
        analysis_input.get("material_model")
        or inp_meta.get("suggested_material_model")
        or "linear_elastic"
    ).strip() or "linear_elastic"
    payload = {
        "schema_version": 1,
        "source_inp": str(inp_meta.get("path") or ""),
        "inp_name": str(inp_meta.get("name") or Path(str(inp_meta.get("path") or "")).name),
        "family": str(inp_meta.get("family") or ""),
        "parser_scope": str(inp_meta.get("parser_scope") or ""),
        "material_name": str(material.get("name") or ""),
        "material_model": material_model,
        "E_override": _optional_float(analysis_input.get("E_override")),
        "nu_override": _optional_float(analysis_input.get("nu_override")),
        "yield_stress_override": _optional_float(analysis_input.get("yield_stress_override")),
        "use_b_ext": _binary_flag(analysis_input.get("use_b_ext"), default=0),
        "gauss_order": _optional_positive_int(analysis_input.get("gauss_order")),
        "n_increments": _positive_int(
            analysis_input.get("n_increments"),
            default=_positive_int(step.get("default_n_increments"), default=1),
        ),
        "max_iterations": _optional_positive_int(analysis_input.get("max_iterations")),
        "convergence_tol": _optional_positive_float(analysis_input.get("convergence_tol")),
        "supports_elastoplastic": bool(inp_meta.get("supports_elastoplastic", False)),
        "parsed_defaults": {
            "material_E": _optional_float(elastic.get("E")),
            "material_nu": _optional_float(elastic.get("nu")),
            "yield_stress": _optional_float(plastic.get("yield_stress")),
            "initial_increment": _optional_float(step.get("initial_increment")),
            "total_time": _optional_float(step.get("total_time")),
            "default_n_increments": _positive_int(step.get("default_n_increments"), default=1),
        },
        "saved_at": _now_iso(),
        "saved_source": str(source).strip() or "web_inc_analysis",
    }
    if material_model != "j2_perfect_plastic":
        payload["yield_stress_override"] = None
    return payload


def _tracked_changes(previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    if not previous:
        return list(_TRACKED_INPUT_KEYS)
    changed = []
    for key in _TRACKED_INPUT_KEYS:
        if previous.get(key) != current.get(key):
            changed.append(key)
    return changed


def _history_entry(snapshot: dict[str, Any], changed_fields: list[str], *, source: str) -> dict[str, Any]:
    return {
        "saved_at": snapshot.get("saved_at") or _now_iso(),
        "source": str(source).strip() or "web_inc_analysis",
        "changed_fields": list(changed_fields),
        "analysis_input": snapshot,
    }


def _merge_metadata(
    existing: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    has_result: bool,
    source: str,
) -> dict[str, str]:
    stage = "result" if has_result else "input_only"
    merged = {str(key): _string_value(value) for key, value in existing.items()}
    merged.update(
        {
            "source_inp": str(snapshot.get("source_inp") or ""),
            "family": str(snapshot.get("family") or merged.get("family") or ""),
            "parser_scope": str(snapshot.get("parser_scope") or ""),
            "material_model": str(snapshot.get("material_model") or "linear_elastic"),
            "configured_use_b_ext": str(snapshot.get("use_b_ext", 0)),
            "configured_gauss_order": ""
            if snapshot.get("gauss_order") is None
            else str(snapshot.get("gauss_order")),
            "configured_n_increments": str(snapshot.get("n_increments", 1)),
            "configured_max_iterations": ""
            if snapshot.get("max_iterations") is None
            else str(snapshot.get("max_iterations")),
            "configured_convergence_tol": ""
            if snapshot.get("convergence_tol") is None
            else str(snapshot.get("convergence_tol")),
            "E_override": "" if snapshot.get("E_override") is None else str(snapshot["E_override"]),
            "nu_override": "" if snapshot.get("nu_override") is None else str(snapshot["nu_override"]),
            "yield_stress_override": ""
            if snapshot.get("yield_stress_override") is None
            else str(snapshot["yield_stress_override"]),
            "analysis_input_saved_at": str(snapshot.get("saved_at") or ""),
            "analysis_input_saved_source": str(source).strip() or "web_inc_analysis",
            "analysis_stage": stage,
            "has_analysis_result": "1" if has_result else "0",
            "analysis_input_sidecar": "0" if has_result else "1",
        }
    )
    return merged


def _load_mat_payload(mat_path: str | Path) -> dict[str, Any]:
    path = Path(mat_path)
    if not path.is_file():
        return {}
    raw = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
    return _normalize_top_level_payload(raw)


def _upsert_analysis_payload(
    mat_path: str | Path,
    inp_meta: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    path = Path(mat_path)
    payload = _load_mat_payload(path)
    existing_history = _parse_json_field(payload, _ANALYSIS_INPUT_HISTORY_JSON_KEY) or []
    if not isinstance(existing_history, list):
        existing_history = []
    previous_input = _parse_json_field(payload, _ANALYSIS_INPUT_JSON_KEY) or {}
    if not isinstance(previous_input, dict):
        previous_input = {}

    changed_fields = _tracked_changes(previous_input, snapshot)
    history = list(existing_history)
    if changed_fields or not history:
        history.append(_history_entry(snapshot, changed_fields, source=source))

    existing_metadata = _struct_to_dict(payload.get("metadata"))
    has_result = _payload_has_analysis_result(payload)
    payload[_INP_METADATA_JSON_KEY] = _json_field(inp_meta)
    payload[_ANALYSIS_INPUT_JSON_KEY] = _json_field(snapshot)
    payload[_ANALYSIS_INPUT_HISTORY_JSON_KEY] = _json_field(history)
    payload["metadata"] = _merge_metadata(existing_metadata, snapshot, has_result=has_result, source=source)

    path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(path), payload, do_compression=True)
    return {
        "path": str(path),
        "changed_fields": changed_fields,
        "history_count": len(history),
        "analysis_stage": "result" if has_result else "input_only",
        "has_analysis_result": has_result,
    }


def save_analysis_input_snapshot(
    result_mat_path: str | Path,
    inp_meta: dict[str, Any],
    analysis_input: dict[str, Any],
    *,
    source: str = "web_inc_analysis",
) -> dict[str, Any]:
    """Save parse metadata and editable analysis input to sidecar MAT.

    The sidecar is temporary: the final result MAT is only updated by the
    exporter after a successful analysis run.
    """
    final_path = Path(result_mat_path)
    snapshot = _build_analysis_input(inp_meta, analysis_input, source=source)
    sidecar_status = _upsert_analysis_payload(
        analysis_input_sidecar_path(final_path),
        inp_meta,
        snapshot,
        source=source,
    )
    return {
        "result_mat_path": str(final_path),
        "sidecar_mat_path": sidecar_status["path"],
        "analysis_input": snapshot,
        "changed_fields": sidecar_status["changed_fields"],
        "history_count": sidecar_status["history_count"],
        "updated_result_mat": False,
    }


def load_analysis_input_sidecar_payload(result_mat_path: str | Path) -> dict[str, Any]:
    """Load the parse-sidecar payload corresponding to one result MAT path."""
    return _load_mat_payload(analysis_input_sidecar_path(result_mat_path))


def decode_analysis_input_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode persisted JSON payloads from a loaded MAT dict."""
    analysis_input = _parse_json_field(payload, _ANALYSIS_INPUT_JSON_KEY)
    history = _parse_json_field(payload, _ANALYSIS_INPUT_HISTORY_JSON_KEY)
    inp_meta = _parse_json_field(payload, _INP_METADATA_JSON_KEY)
    return {
        "analysis_input": analysis_input if isinstance(analysis_input, dict) else None,
        "analysis_input_history": history if isinstance(history, list) else [],
        "inp_metadata": inp_meta if isinstance(inp_meta, dict) else None,
    }


__all__ = [
    "analysis_input_sidecar_path",
    "delete_analysis_input_sidecar",
    "decode_analysis_input_bundle",
    "load_analysis_input_sidecar_payload",
    "save_analysis_input_snapshot",
]