"""Lightweight MAT reader for demo elastic and shakedown visualization."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from jaxmech.model.viz_manifest import extract_viz_manifest


@dataclass
class FieldInfo:
    key: str
    label: str
    n_components: int
    location: str
    n_frames: int = 1
    symbol: str = ""
    formula: str = ""
    description: str = ""

    @property
    def is_multiframe(self) -> bool:
        return self.n_frames > 1


@dataclass
class VizData:
    points: Optional[np.ndarray] = None
    cells: Optional[np.ndarray] = None
    cell_types: Optional[np.ndarray] = None
    cell_ele_types: Optional[np.ndarray] = None
    fields: dict[str, np.ndarray] = dc_field(default_factory=dict)
    field_info: dict[str, FieldInfo] = dc_field(default_factory=dict)
    frame_times: Optional[np.ndarray] = None
    n_gauss_per_elem: Optional[np.ndarray] = None
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    source_path: str = ""

    @property
    def has_mesh(self) -> bool:
        return self.points is not None and self.cells is not None and self.cell_types is not None

    def field_keys(self) -> list[str]:
        return list(self.field_info.keys())


_ABAQUS_TO_VTK = {
    "C3D4": 10,
    "C3D10": 24,
    "C3D6": 13,
    "C3D8": 12,
    "C3D8R": 12,
    "C3D20": 25,
    "C3D20R": 25,
}


def _clean_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in raw.items() if not str(key).startswith("__")}


def _unwrap(value: Any) -> Any:
    cur = value
    for _ in range(8):
        if isinstance(cur, np.ndarray) and cur.size == 1 and (cur.dtype == object or not getattr(cur.dtype, "names", None)):
            cur = cur.reshape(-1)[0]
            continue
        break
    return cur


def _struct_get(obj: Any, key: str) -> Any:
    obj = _unwrap(obj)
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        values = [_struct_get(item, key) for item in obj.flat]
        values = [item for item in values if item is not None]
        return values[0] if values else None
    if isinstance(obj, np.ndarray) and getattr(obj.dtype, "names", None) and key in obj.dtype.names:
        return obj[key]
    if hasattr(obj, key):
        return getattr(obj, key)
    return None


def _records(obj: Any) -> list[Any]:
    obj = _unwrap(obj)
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        if obj.dtype == object or getattr(obj.dtype, "names", None):
            return [item for item in obj.flat if item is not None]
    return [obj]


def _best_result_record(raw: Mapping[str, Any]) -> Any:
    records = _records(raw.get("ResultsSet"))
    if not records:
        return None
    best = records[0]
    best_alpha = -np.inf
    for rec in records:
        value = _struct_get(rec, "objective_value")
        try:
            alpha = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
        except Exception:
            alpha = -np.inf
        if alpha >= best_alpha:
            best = rec
            best_alpha = alpha
    return best


def _lookup(raw: Mapping[str, Any], key: str) -> Any:
    if key in raw:
        return raw[key]
    for parent in ("ElasticInputSet", "ConfigInfo"):
        value = _struct_get(raw.get(parent), key)
        if value is not None:
            return value
    best = _best_result_record(raw)
    value = _struct_get(best, key)
    return value


def _text(value: Any) -> str:
    cur = _unwrap(value)
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore").strip()
    if isinstance(cur, np.ndarray):
        if cur.size == 0:
            return ""
        return _text(cur.reshape(-1)[0])
    return str(cur).strip() if cur is not None else ""


def _as_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return np.squeeze(arr)


def _as_int_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.int64)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return np.squeeze(arr)


def _as_object_list(value: Any) -> list[Any]:
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [item for item in value.reshape(-1)]
    return [value]


def _load_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    meta = _unwrap(raw.get("metadata"))
    if isinstance(meta, Mapping):
        return {str(key): _text(value) for key, value in meta.items()}
    if hasattr(meta, "_fieldnames"):
        return {str(name): _text(getattr(meta, name)) for name in meta._fieldnames}
    return {}


def _pack_cells(blocks: list[np.ndarray]) -> np.ndarray:
    packed: list[int] = []
    for block in blocks:
        arr = np.asarray(block, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size and arr.min() >= 1:
            arr = arr - 1
        for conn in arr:
            packed.append(int(conn.size))
            packed.extend(int(i) for i in conn)
    return np.asarray(packed, dtype=np.int64)


def _load_mesh(raw: Mapping[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    points = _as_float_array(raw.get("viz_points"))
    if points is None:
        points = _as_float_array(raw.get("points"))
    cells = _as_int_array(raw.get("viz_cells"))
    cell_types = _as_int_array(raw.get("viz_cell_types"))
    ele_types_value = raw.get("viz_cell_block_ele_types")
    if ele_types_value is None:
        ele_types_value = raw.get("solid_elem_types")

    inp_data = _unwrap(raw.get("InpData"))
    if points is None and inp_data is not None:
        points = _as_float_array(_struct_get(inp_data, "points"))
    if cells is None and inp_data is not None:
        blocks = [_as_int_array(item) for item in _as_object_list(_struct_get(inp_data, "cell_block_cells"))]
        blocks = [block for block in blocks if block is not None]
        if blocks:
            cells = _pack_cells(blocks)
    if ele_types_value is None and inp_data is not None:
        ele_types_value = _struct_get(inp_data, "cell_block_ele_types")

    ele_types = [_text(item).upper() for item in _as_object_list(ele_types_value)]
    if cell_types is None and ele_types and cells is not None:
        per_cell: list[int] = []
        pos = 0
        i = 0
        while pos < int(cells.size):
            per_cell.append(int(_ABAQUS_TO_VTK.get(ele_types[min(i, len(ele_types) - 1)], 12)))
            pos += 1 + int(cells[pos])
            i += 1
        cell_types = np.asarray(per_cell, dtype=np.int32)
    if points is not None and points.ndim == 1:
        points = points.reshape(-1, 3)
    if points is not None and points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1), dtype=points.dtype)])
    return points, cells, cell_types, np.asarray(ele_types, dtype=object) if ele_types else None


def _manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    manifest = extract_viz_manifest(raw)
    if manifest:
        return manifest
    if "ResultsSet" in raw or "ElasticInputSet" in raw:
        from jaxmech.modules.shakedown.visualize_mat import build_shakedown_viz_manifest

        return build_shakedown_viz_manifest(raw)
    from jaxmech.modules.inc_analysis.visualize_mat import build_solid_elastic_viz_manifest

    return build_solid_elastic_viz_manifest(raw)


def _stack_result_field(raw: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    frames = []
    for rec in _records(raw.get("ResultsSet")):
        arr = _as_float_array(_struct_get(rec, key))
        if arr is not None:
            frames.append(arr)
    if not frames:
        return _as_float_array(_lookup(raw, key))
    try:
        return np.stack(frames, axis=0)
    except Exception:
        return frames[0]


def _objective_values(raw: Mapping[str, Any]) -> np.ndarray:
    values = []
    for rec in _records(raw.get("ResultsSet")):
        val = _as_float_array(_struct_get(rec, "objective_value"))
        if val is not None:
            values.append(float(val.reshape(-1)[0]))
    if not values:
        val = _as_float_array(_lookup(raw, "objective_value"))
        values.append(float(val.reshape(-1)[0]) if val is not None else 1.0)
    return np.asarray(values, dtype=np.float64)


def _load_factors(raw: Mapping[str, Any], n_cases: int) -> np.ndarray:
    factors = []
    for rec in _records(raw.get("ResultsSet")):
        val = _as_float_array(_struct_get(rec, "load_factor"))
        if val is not None:
            factors.append(val.reshape(-1)[:n_cases])
    if not factors:
        val = _as_float_array(_lookup(raw, "load_factor"))
        if val is not None:
            factors.append(val.reshape(-1)[:n_cases])
    if not factors:
        factors.append(np.ones((n_cases,), dtype=np.float64))
    return np.asarray(factors, dtype=np.float64)


def _gauss_stress_cases(raw: Mapping[str, Any]) -> Optional[np.ndarray]:
    arr = _as_float_array(_lookup(raw, "gauss_stress_cases"))
    if arr is None:
        arr = _as_float_array(_lookup(raw, "sigma_E"))
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def _field_array(raw: Mapping[str, Any], field: Mapping[str, Any]) -> Optional[np.ndarray]:
    key = str(field.get("key") or "")
    source_key = str(field.get("source_key") or key)
    if key == "shakedown_residual_stress":
        return _as_float_array(_lookup(raw, "residual_stress"))
    if key == "shakedown_inequality_violation":
        ratio = _stack_result_field(raw, "yield_ratio")
        return None if ratio is None else np.maximum(np.asarray(ratio, dtype=np.float64) - 1.0, 0.0)
    if key == "shakedown_total_stress":
        residual = _as_float_array(_lookup(raw, "residual_stress"))
        cases = _gauss_stress_cases(raw)
        if residual is None or cases is None:
            return None
        n_cases = int(cases.shape[2])
        factors = _load_factors(raw, n_cases)
        alphas = _objective_values(raw)
        frames = []
        for i, factor in enumerate(factors):
            sigma_e = np.tensordot(cases, factor[:n_cases], axes=([2], [0]))
            alpha = float(alphas[min(i, alphas.size - 1)])
            frames.append(np.asarray(residual, dtype=np.float64) + alpha * sigma_e)
        return np.stack(frames, axis=0)
    if key == "sigma_E":
        cases = _gauss_stress_cases(raw)
        return None if cases is None else cases[:, :, 0]
    return _as_float_array(_lookup(raw, source_key))


def _field_info_from_entry(key: str, entry: Mapping[str, Any], arr: np.ndarray) -> FieldInfo:
    n_components = int(entry.get("n_components") or 1)
    if arr.ndim >= 2 and int(arr.shape[-1]) in (1, 2, 3, 4, 6):
        n_components = int(arr.shape[-1])
    frames_decl = str(entry.get("frames") or "single")
    n_frames = int(arr.shape[0]) if frames_decl == "frames" and arr.ndim >= 2 else 1
    return FieldInfo(
        key=key,
        label=str(entry.get("label") or key),
        n_components=n_components,
        location=str(entry.get("location") or "node"),
        n_frames=max(1, n_frames),
        symbol=str(entry.get("symbol") or ""),
        formula=str(entry.get("formula") or ""),
        description=str(entry.get("description") or ""),
    )


def inspect_mat_fields(mat_path: str | Path) -> dict:
    raw = _clean_raw(__import__("scipy.io").io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False))
    manifest = _manifest(raw)
    fields = []
    for entry in manifest.get("fields", []):
        key = str(entry.get("key") or "")
        if not key:
            continue
        arr = _field_array(raw, entry)
        if arr is None:
            continue
        info = _field_info_from_entry(key, entry, np.asarray(arr))
        fields.append(
            {
                "key": info.key,
                "label": info.label,
                "n_components": info.n_components,
                "location": info.location,
                "n_frames": info.n_frames,
                "symbol": info.symbol,
                "formula": info.formula,
                "description": info.description,
            }
        )
    return {"mat_path": str(mat_path), "fields": fields, "manifest": manifest}


def load_viz_data(
    mat_path: str | Path,
    *,
    selected_fields: Optional[Sequence[str]] = None,
) -> VizData:
    import scipy.io as sio

    path = Path(mat_path)
    raw = _clean_raw(sio.loadmat(str(path), squeeze_me=False, struct_as_record=False))
    points, cells, cell_types, ele_types = _load_mesh(raw)
    manifest = _manifest(raw)
    selected = {str(item) for item in selected_fields or [] if str(item).strip()}

    vd = VizData(
        points=points,
        cells=cells,
        cell_types=cell_types,
        cell_ele_types=ele_types,
        n_gauss_per_elem=_as_int_array(raw.get("n_gauss_per_elem")),
        metadata={**_load_metadata(raw), "manifest": manifest},
        source_path=str(path),
    )

    for entry in manifest.get("fields", []):
        key = str(entry.get("key") or "")
        if not key or (selected and key not in selected):
            continue
        arr = _field_array(raw, entry)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        vd.fields[key] = arr
        vd.field_info[key] = _field_info_from_entry(key, entry, arr)

    if not vd.fields and selected:
        for key in selected:
            arr = _as_float_array(_lookup(raw, key))
            if arr is None:
                continue
            vd.fields[key] = arr
            vd.field_info[key] = FieldInfo(key=key, label=key, n_components=arr.shape[-1] if arr.ndim >= 2 else 1, location="node")

    return vd
