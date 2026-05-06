"""A/B visual comparison helpers for the Web visualization module."""

from __future__ import annotations

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.io as sio

from jaxmech.modules.visualization.mesh_builder import scalarize_field

DIFF_FIELD_KEY = "viz_diff"


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite scalar values available for comparison.")
    q1, median, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    inlier = arr[(arr >= low_fence) & (arr <= high_fence)]
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(arr.max()),
        "iqr": float(iqr),
        "whisker_low": float(inlier.min()) if inlier.size else float(arr.min()),
        "whisker_high": float(inlier.max()) if inlier.size else float(arr.max()),
        "outlier_count": int(arr.size - inlier.size),
    }


def _iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _file_info(path_text: str) -> dict[str, Any]:
    path = Path(path_text) if path_text else None
    info: dict[str, Any] = {
        "path": str(path) if path else "",
        "name": path.name if path else "",
        "stem": path.stem if path else "",
    }
    if path and path.exists():
        stat = path.stat()
        info.update({
            "ctime": float(stat.st_ctime),
            "ctime_iso": _iso(stat.st_ctime),
            "mtime": float(stat.st_mtime),
            "mtime_iso": _iso(stat.st_mtime),
        })
    return info


def _selection(scene) -> dict[str, Any]:
    info = scene.info
    fi = scene.active_field_info()
    return {
        "source": info.get("source", ""),
        "field": info.get("active_field", ""),
        "field_label": info.get("active_field_label", ""),
        "scalar_label": info.get("active_scalar_label", ""),
        "component": info.get("active_component"),
        "frame": int(info.get("active_frame", 0) or 0),
        "n_frames": int(info.get("n_frames", 0) or 0),
        "location": fi.location if fi else "",
        "n_components": fi.n_components if fi else 0,
        "is_multiframe": bool(fi.is_multiframe) if fi else False,
    }


def _require_same_mesh(vd_a, vd_b) -> None:
    if vd_a is None or vd_b is None or vd_a.points is None or vd_b.points is None:
        return
    pts_a = np.asarray(vd_a.points, dtype=np.float64)
    pts_b = np.asarray(vd_b.points, dtype=np.float64)
    if pts_a.shape != pts_b.shape:
        raise ValueError(f"Meshes are not node-compatible: {pts_a.shape} vs {pts_b.shape}.")
    if not np.allclose(pts_a, pts_b, rtol=1.0e-8, atol=1.0e-10, equal_nan=True):
        raise ValueError("Meshes have different node coordinates; Diff requires matching displayed nodes.")


def _same_frame_axis(scene_a, scene_b, n_frames: int) -> bool:
    vd_a = getattr(scene_a, "_vd", None)
    vd_b = getattr(scene_b, "_vd", None)
    times_a = getattr(vd_a, "frame_times", None)
    times_b = getattr(vd_b, "frame_times", None)
    if times_a is None or times_b is None:
        return True
    arr_a = np.asarray(times_a, dtype=np.float64).reshape(-1)
    arr_b = np.asarray(times_b, dtype=np.float64).reshape(-1)
    if arr_a.size < n_frames or arr_b.size < n_frames:
        return False
    return bool(np.allclose(arr_a[:n_frames], arr_b[:n_frames], rtol=1.0e-8, atol=1.0e-10, equal_nan=True))


def _diff(values_b: np.ndarray, values_a: np.ndarray, *, absolute: bool) -> np.ndarray:
    out = np.asarray(values_b, dtype=np.float64) - np.asarray(values_a, dtype=np.float64)
    return np.abs(out) if absolute else out


def _diff_filename(metadata: dict[str, Any]) -> str:
    mat_a = metadata.get("mat_a", {}) if isinstance(metadata.get("mat_a"), dict) else {}
    mat_b = metadata.get("mat_b", {}) if isinstance(metadata.get("mat_b"), dict) else {}
    stem_a = str(mat_a.get("stem") or "A")
    stem_b = str(mat_b.get("stem") or "B")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem_a}__vs__{stem_b}__diff_{stamp}.mat"


def _as_numeric(value: Any) -> np.ndarray:
    return np.squeeze(np.asarray(value, dtype=np.float64))


def _load_original_field(selection: dict[str, Any], vd) -> np.ndarray:
    """Load the selected variable from its original MAT file."""
    key = str(selection.get("field") or "")
    source = str(selection.get("source") or "")
    if key and source:
        path = Path(source)
        if path.exists():
            raw = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False, variable_names=[key])
            if key in raw:
                return np.asarray(raw[key])
    if vd is not None and key in vd.fields:
        return np.asarray(vd.fields[key])
    raise ValueError(f"Cannot find original MAT variable: {key}")


def _scalarize_original(raw: np.ndarray, selection: dict[str, Any], vd, *, frame: Optional[int] = None) -> np.ndarray:
    arr = _as_numeric(raw)
    if bool(selection.get("is_multiframe", False)):
        idx = int(selection.get("frame", 0) if frame is None else frame)
        idx = max(0, min(idx, int(selection.get("n_frames", 1) or 1) - 1))
        if arr.ndim == 0:
            raise ValueError(f"Frame variable has no frame axis: {selection.get('field')}")
        arr = arr[idx]
    return scalarize_field(
        arr,
        vd,
        location=str(selection.get("location") or "node"),
        component=selection.get("component"),
        field_key=str(selection.get("field") or ""),
    )


def _signed_diff_from_original(
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    metadata: dict[str, Any],
    vd_a,
    vd_b,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute signed A-B Diff from original MAT variables."""
    sel_a = metadata.get("selection_a", {})
    sel_b = metadata.get("selection_b", {})
    full_frame = bool(metadata.get("full_frame", False))
    if full_frame:
        n_frames = min(int(sel_a.get("n_frames", 0) or 0), int(sel_b.get("n_frames", 0) or 0))
        frames = []
        for idx in range(n_frames):
            a = _scalarize_original(raw_a, sel_a, vd_a, frame=idx)
            b = _scalarize_original(raw_b, sel_b, vd_b, frame=idx)
            if a.shape != b.shape:
                raise ValueError(f"Original scalar fields are not shape-compatible at frame {idx}: {a.shape} vs {b.shape}")
            frames.append(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        frame_diff = np.stack(frames, axis=0)
        active = max(0, min(int(sel_a.get("frame", 0) or 0), frame_diff.shape[0] - 1))
        return frame_diff[active], frame_diff

    a = _scalarize_original(raw_a, sel_a, vd_a)
    b = _scalarize_original(raw_b, sel_b, vd_b)
    if a.shape != b.shape:
        raise ValueError(f"Original scalar fields are not shape-compatible: {a.shape} vs {b.shape}")
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), None


def _component_info(selection: dict[str, Any]) -> dict[str, Any]:
    component = selection.get("component")
    return {
        "variable": str(selection.get("field") or ""),
        "component": int(component) if component is not None else -1,
        "component_label": str(component) if component is not None else "auto",
        "scalar_label": str(selection.get("scalar_label") or ""),
        "frame": int(selection.get("frame", 0) or 0),
        "n_frames": int(selection.get("n_frames", 0) or 0),
        "is_frame_variable": bool(selection.get("is_multiframe", False)),
    }


def _diff_info_struct(metadata: dict[str, Any]) -> dict[str, Any]:
    mat_a = metadata.get("mat_a", {}) if isinstance(metadata.get("mat_a"), dict) else {}
    mat_b = metadata.get("mat_b", {}) if isinstance(metadata.get("mat_b"), dict) else {}
    return {
        "created_at": str(metadata.get("created_at") or ""),
        "formula": "A-B",
        "display_formula": str(metadata.get("formula") or ""),
        "display_abs_selected": bool(metadata.get("absolute", False)),
        "full_frame": bool(metadata.get("full_frame", False)),
        "value_location": str(metadata.get("value_location") or ""),
        "value_semantics": str(metadata.get("value_semantics") or ""),
        "mat_a_name": str(mat_a.get("name") or ""),
        "mat_a_path": str(mat_a.get("path") or ""),
        "mat_a_ctime": str(mat_a.get("ctime_iso") or ""),
        "mat_a_mtime": str(mat_a.get("mtime_iso") or ""),
        "mat_b_name": str(mat_b.get("name") or ""),
        "mat_b_path": str(mat_b.get("path") or ""),
        "mat_b_ctime": str(mat_b.get("ctime_iso") or ""),
        "mat_b_mtime": str(mat_b.get("mtime_iso") or ""),
        "field_a": _component_info(metadata.get("selection_a", {})),
        "field_b": _component_info(metadata.get("selection_b", {})),
    }


def _diff_mat_payload(
    *,
    diff: np.ndarray,
    frame_diff: Optional[np.ndarray],
    field_a: np.ndarray,
    field_b: np.ndarray,
    metadata: dict[str, Any],
    vd_a,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Diff": np.asarray(diff, dtype=np.float64),
        "Field_A": np.asarray(field_a),
        "Field_B": np.asarray(field_b),
        "DiffInfo": _diff_info_struct(metadata),
    }
    if frame_diff is not None:
        payload["frame_Diff"] = np.asarray(frame_diff, dtype=np.float64)
    if vd_a is not None:
        if vd_a.points is not None:
            payload["points"] = np.asarray(vd_a.points)
        if vd_a.frame_times is not None and frame_diff is not None:
            payload["frame_time"] = np.asarray(vd_a.frame_times)
    return payload


def _diff_mat_bytes(*, diff: np.ndarray, frame_diff: Optional[np.ndarray], field_a: np.ndarray, field_b: np.ndarray, metadata: dict[str, Any], vd_a) -> bytes:
    buf = io.BytesIO()
    sio.savemat(
        buf,
        _diff_mat_payload(
            diff=diff,
            frame_diff=frame_diff,
            field_a=field_a,
            field_b=field_b,
            metadata=metadata,
            vd_a=vd_a,
        ),
        do_compression=True,
    )
    return buf.getvalue()


def compute_session_diff(
    session,
    *,
    absolute: bool = False,
    output_mat: bool = False,
) -> dict[str, Any]:
    """Compute displayed scalar Diff for the session's A/B scenes.

    ``output_mat`` is kept for backward-compatible callers but is intentionally
    ignored.  The Web workflow saves Diff through ``registered_diff_mat`` so the
    browser download location matches screenshot downloads.
    """
    scene_a = session.get_scene("a")
    scene_b = session.get_scene("b")
    if scene_a is None or scene_b is None:
        raise ValueError("Compare mode requires both scene A and scene B to be loaded.")
    if scene_a.info.get("active_field") == DIFF_FIELD_KEY or scene_b.info.get("active_field") == DIFF_FIELD_KEY:
        raise ValueError("Diff cannot be used as an input for another Diff. Select original result fields before comparing.")

    vd_a = session.get_vd("a")
    vd_b = session.get_vd("b")
    _require_same_mesh(vd_a, vd_b)

    values_a = scene_a.get_scalar_data()
    values_b = scene_b.get_scalar_data()
    if values_a.shape != values_b.shape:
        raise ValueError(f"Current scalar fields are not shape-compatible: {values_a.shape} vs {values_b.shape}.")

    current_diff = _diff(values_b, values_a, absolute=absolute)
    sel_a = _selection(scene_a)
    sel_b = _selection(scene_b)
    formula = "|B - A|" if absolute else "B - A"

    full_frame = False
    frame_diff = None
    full_frame_reason = "current displayed frame only"
    series_a = scene_a.get_scalar_frame_series()
    series_b = scene_b.get_scalar_frame_series()
    if (
        series_a is not None
        and series_b is not None
        and sel_a["frame"] == sel_b["frame"]
        and series_a.shape == series_b.shape
        and _same_frame_axis(scene_a, scene_b, int(series_a.shape[0]))
    ):
        frame_diff = _diff(series_b, series_a, absolute=absolute)
        full_frame = True
        full_frame_reason = "all matching frames"

    display_text = (
        f"Diff = {formula}; "
        f"A {sel_a['field']} / {sel_a['scalar_label']} / frame {sel_a['frame'] + 1}; "
        f"B {sel_b['field']} / {sel_b['scalar_label']} / frame {sel_b['frame'] + 1}"
    )
    overlay_formula = "abs(B - A)" if absolute else "B - A"
    overlay_text = (
        f"Diff = {overlay_formula}\n"
        f"A: {sel_a['field']} / {sel_a['scalar_label']} / frame {sel_a['frame'] + 1}\n"
        f"B: {sel_b['field']} / {sel_b['scalar_label']} / frame {sel_b['frame'] + 1}"
    )
    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formula": formula,
        "absolute": bool(absolute),
        "full_frame": bool(full_frame),
        "full_frame_reason": full_frame_reason,
        "field_key": DIFF_FIELD_KEY,
        "value_location": "node",
        "value_semantics": "displayed scalar values after component/equivalent/magnitude selection and any Gauss-to-node averaging",
        "selection_a": sel_a,
        "selection_b": sel_b,
        "mat_a": _file_info(sel_a["source"]),
        "mat_b": _file_info(sel_b["source"]),
        "diff_shape": list(np.asarray(current_diff).shape),
        "frame_diff_shape": list(np.asarray(frame_diff).shape) if frame_diff is not None else None,
        "display_text": display_text,
        "overlay_text": overlay_text,
    }

    field_values = frame_diff if frame_diff is not None else current_diff
    n_frames = int(field_values.shape[0]) if frame_diff is not None else 1
    active_frame = sel_a["frame"] if frame_diff is not None else 0
    for slot, set_active in (("a", True), ("b", False)):
        scene = session.get_scene(slot)
        if scene is None:
            continue
        scene.register_scalar_field(
            DIFF_FIELD_KEY,
            field_values,
            label="Diff",
            n_frames=n_frames,
            metadata=metadata,
            set_active=set_active,
            frame=active_frame,
        )

    return {
        "ok": True,
        "label": formula,
        "absolute": bool(absolute),
        "diff_key": DIFF_FIELD_KEY,
        "diff_info": metadata,
        "diff_info_text": display_text,
        "selection_a": sel_a,
        "selection_b": sel_b,
        "stats": _finite_stats(current_diff),
        "full_frame": bool(full_frame),
        "scene_a": scene_a.info,
        "scene_b": scene_b.info,
    }


def registered_diff_summary(session) -> dict[str, Any]:
    """Return boxplot-ready metadata for the registered Diff field."""
    scene_a = session.get_scene("a")
    if scene_a is None:
        raise ValueError("No scene A is loaded.")
    vd_a = session.get_vd("a")
    if vd_a is None or DIFF_FIELD_KEY not in vd_a.fields:
        raise ValueError("No Diff field is available. Run compare first.")
    metadata = vd_a.metadata.get("generated_field_info", {}).get(DIFF_FIELD_KEY, {})
    values = np.asarray(vd_a.fields[DIFF_FIELD_KEY], dtype=np.float64)
    if values.ndim >= 2:
        frame = int(metadata.get("selection_a", {}).get("frame", scene_a.info.get("active_frame", 0)) or 0)
        frame = max(0, min(frame, values.shape[0] - 1))
        current = values[frame]
    else:
        current = values
    return {
        "ok": True,
        "label": metadata.get("formula", "Diff"),
        "absolute": bool(metadata.get("absolute", False)),
        "diff_key": DIFF_FIELD_KEY,
        "diff_info": metadata,
        "diff_info_text": metadata.get("display_text", ""),
        "selection_a": metadata.get("selection_a", {}),
        "selection_b": metadata.get("selection_b", {}),
        "stats": _finite_stats(current),
        "full_frame": bool(metadata.get("full_frame", False)),
        "scene_a": scene_a.info,
        "scene_b": session.get_scene("b").info if session.get_scene("b") else None,
    }


def registered_diff_mat(session) -> tuple[str, bytes]:
    """Return filename and MAT bytes for the registered Diff field."""
    scene_a = session.get_scene("a")
    scene_b = session.get_scene("b")
    vd_a = session.get_vd("a")
    vd_b = session.get_vd("b")
    if scene_a is None or scene_b is None or vd_a is None or vd_b is None or DIFF_FIELD_KEY not in vd_a.fields:
        raise ValueError("No Diff field is available. Run compare first.")
    metadata = vd_a.metadata.get("generated_field_info", {}).get(DIFF_FIELD_KEY, {})
    sel_a = metadata.get("selection_a", {})
    sel_b = metadata.get("selection_b", {})
    raw_a = _load_original_field(sel_a, vd_a)
    raw_b = _load_original_field(sel_b, vd_b)
    diff, frame_diff = _signed_diff_from_original(raw_a, raw_b, metadata, vd_a, vd_b)
    return _diff_filename(metadata), _diff_mat_bytes(
        diff=diff,
        frame_diff=frame_diff,
        field_a=raw_a,
        field_b=raw_b,
        metadata=metadata,
        vd_a=vd_a,
    )
