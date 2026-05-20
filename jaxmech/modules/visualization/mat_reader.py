"""Unified MAT reader for the visualization module.

Reads any jaxmech-produced .mat file and extracts a standardised
``VizData`` dict suitable for building a PyVista scene.  Handles:

* AnalysisResult .mat (``to_mat`` output)
* DCA result .mat (``frame_*`` arrays + optional ``InpData``)
* RSDM result .mat (``residual_stress_cycle`` etc.)
* InpData-only .mat (mesh geometry, no field data)

The reader never imports ``Model`` / ``AnalysisResult``; it works
directly on the scipy.io.loadmat dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from jaxmech.io.mat_schema import (
    FRAME_OUTPUTS_KEY,
    SHELL_PARAM_KEY,
    SOLID_PARAM_KEY,
    VALIDATION_PARAM_KEY,
    VALIDATION_RESULT_KEY,
    expand_public_mat_payload,
    nested_frame_output_shapes,
    nested_struct_shapes,
)
from jaxmech.model.viz_manifest import (
    VIZ_MANIFEST_KEY,
    extract_viz_manifest,
    load_viz_manifest_from_mat,
    write_viz_manifest_to_mat,
)

# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass
class FieldInfo:
    """Descriptor for a single visualisable field."""

    key: str
    label: str
    n_components: int
    location: str  # "gauss", "node", or "element"
    n_frames: int = 1
    symbol: str = ""
    formula: str = ""
    description: str = ""

    @property
    def is_multiframe(self) -> bool:
        return self.n_frames > 1


@dataclass
class VizData:
    """Everything the visualisation backend needs from a .mat file."""

    # Mesh geometry
    points: Optional[np.ndarray] = None        # (n_nodes, 2|3)
    cells: Optional[np.ndarray] = None         # VTK-packed 1-D
    cell_types: Optional[np.ndarray] = None    # per-element VTK type id
    cell_ele_types: Optional[np.ndarray] = None  # per-element ABAQUS name

    # Discovered field variables: key -> array
    fields: Dict[str, np.ndarray] = dc_field(default_factory=dict)

    # Field metadata
    field_info: Dict[str, FieldInfo] = dc_field(default_factory=dict)

    # Frame time axis (if any)
    frame_times: Optional[np.ndarray] = None

    # Gauss-to-element bookkeeping
    n_gauss_per_elem: Optional[np.ndarray] = None

    # Raw metadata dict from the .mat
    metadata: Dict[str, Any] = dc_field(default_factory=dict)

    # Source path for display
    source_path: str = ""

    @property
    def has_mesh(self) -> bool:
        return self.points is not None and self.cells is not None

    def field_keys(self) -> List[str]:
        return list(self.field_info.keys())


# ---------------------------------------------------------------------------
# ABAQUS -> VTK mapping (mirrors _ABAQUS_TO_VTK in result.py)
# ---------------------------------------------------------------------------

_ABAQUS_TO_VTK: dict[str, int] = {
    "C3D4": 10, "C3D10": 24, "C3D6": 13,
    "C3D8": 12, "C3D8R": 12, "C3D8I": 12,
    "C3D20": 25, "C3D20R": 25,
    "CPE3": 5,  "CPS3": 5,
    "S3": 5,    "S3R": 5,   "STRI3": 5,
    "CPE4": 9,  "CPE4R": 9, "CPS4": 9, "CPS4R": 9,
    "S4": 9,    "S4R": 9,
    "CPE6": 22, "CPS6": 22,
    "CPE8": 23, "CPS8": 23,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _squeeze(arr: Any) -> np.ndarray:
    """Convert MATLAB-loaded array to a sensible numpy array."""
    a = np.asarray(arr)
    while a.ndim > 0 and a.shape[0] == 1 and a.ndim > 1:
        a = a[0]
    return np.squeeze(a)


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap MATLAB scalar/object arrays without disturbing real data arrays."""
    cur = value
    for _ in range(8):
        if isinstance(cur, np.ndarray) and cur.size == 1 and not getattr(cur.dtype, "names", None):
            cur = cur.flat[0]
            continue
        break
    return cur


def _struct_get(obj: Any, key: str) -> Any:
    """Read a field from a scipy-loaded MATLAB struct or a plain dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        values = []
        for item in obj.flat:
            value = _struct_get(item, key)
            if value is not None:
                values.append(value)
        if not values:
            return None
        return values[0] if len(values) == 1 else values
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        if key not in obj.dtype.names:
            return None
        src = obj.flat[0] if getattr(obj, "size", 0) == 1 else obj
        return src[key]
    return None


def _resultset_records(obj: Any) -> list[Any]:
    """Return individual structs from a Version C ``ResultsSet`` container."""
    if obj is None:
        return []
    records: list[Any] = []
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for item in obj.flat:
                if isinstance(item, np.ndarray) and getattr(item.dtype, "names", None):
                    records.extend(list(item.flat))
                else:
                    records.append(item)
        elif getattr(obj.dtype, "names", None):
            records.extend(list(obj.flat))
        else:
            records.append(obj)
    else:
        records.append(obj)
    return [item for item in records if item is not None]


def _resultset_get(obj: Any, key: str) -> Any:
    """Read a field from a Version C ResultsSet cell array.

    Multi-parameter Gurobi MAT files store one result struct per parameter set.
    For visualization we pick the row with the largest objective value, matching
    the summary convention used elsewhere in the shakedown workflow.
    """
    if obj is None:
        return None

    values: list[Any] = []
    objectives: list[float] = []

    for item in _resultset_records(obj):
        value = _struct_get(item, key)
        if value is None:
            continue
        objective = _struct_get(item, "objective_value")
        scalar = _alpha_scalar(objective)
        if isinstance(value, list):
            for sub in value:
                values.append(sub)
                if scalar is not None:
                    objectives.append(float(scalar))
        else:
            values.append(value)
            if scalar is not None:
                objectives.append(float(scalar))

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if objectives and len(objectives) == len(values):
        return values[int(np.nanargmax(np.asarray(objectives, dtype=np.float64)))]
    return values[0]


def _lookup_any(raw: dict, key: str, result_record: Any = None) -> Any:
    """Look up a top-level or shakedown-nested MAT field."""
    if key in raw:
        return raw.get(key)
    for parent in ("ElasticInputSet", "ConfigInfo", "RSDMResult", "RSDMInput"):
        value = _struct_get(raw.get(parent), key)
        if value is not None:
            return value
    if result_record is not None:
        value = _struct_get(result_record, key)
        if value is not None:
            return value
    value = _resultset_get(raw.get("ResultsSet"), key)
    if value is not None:
        return value
    return None


def _as_float_array(value: Any, *, squeeze: bool = True) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if squeeze:
        arr = _squeeze(arr)
    if arr.size == 0:
        return None
    return arr


def _extract_mesh_from_viz_keys(raw: dict) -> tuple:
    """Extract mesh from ``viz_*`` keys written by ``_mesh_to_viz_dict``."""
    pts = raw.get("viz_points")
    cells = raw.get("viz_cells")
    ctypes = raw.get("viz_cell_types")
    eletypes = raw.get("viz_cell_block_ele_types")
    if pts is not None and cells is not None and ctypes is not None:
        return (
            _squeeze(pts).astype(np.float64),
            _squeeze(cells).astype(np.int64),
            _squeeze(ctypes).astype(np.int32),
            _squeeze(eletypes) if eletypes is not None else None,
        )
    return None, None, None, None


def _extract_mesh_from_inpdata(raw: dict) -> tuple:
    """Extract mesh from nested ``InpData`` struct (DCA format)."""
    inp = raw.get("InpData")
    if inp is None:
        return None, None, None, None

    # scipy struct record
    if hasattr(inp, "dtype") and getattr(inp.dtype, "names", None):
        src = inp.flat[0] if inp.size == 1 else inp
        names = set(inp.dtype.names or ())

        if {"points", "cell_block_cells"}.issubset(names):
            try:
                points = np.asarray(src["points"], dtype=np.float64)
                points = _squeeze(points)
            except Exception:
                points = None

            try:
                ctn = src["cell_block_cell_types"]
                ctn = [str(np.asarray(x).flat[0]) for x in np.asarray(ctn).flat]
            except Exception:
                ctn = []

            try:
                etn = src["cell_block_ele_types"]
                etn = [str(np.asarray(x).flat[0]) for x in np.asarray(etn).flat]
            except Exception:
                etn = ctn

            try:
                cblocks = src["cell_block_cells"]
                cblocks = [np.asarray(x, dtype=np.int64) for x in np.asarray(cblocks).flat]
            except Exception:
                cblocks = []

            all_cells_parts: list[np.ndarray] = []
            all_types: list[int] = []
            all_eletypes: list[str] = []

            for idx, conn_raw in enumerate(cblocks):
                conn = _squeeze(conn_raw).astype(np.int64)
                if conn.ndim == 1:
                    continue
                n_elem, npe = conn.shape
                ele_key = etn[idx].upper() if idx < len(etn) else ""
                vtk_type = _ABAQUS_TO_VTK.get(ele_key, 12)
                prefix = np.full((n_elem, 1), npe, dtype=np.int64)
                all_cells_parts.append(np.hstack([prefix, conn]))
                all_types.extend([vtk_type] * n_elem)
                all_eletypes.extend([ele_key] * n_elem)

            if points is not None and all_cells_parts:
                cells = np.concatenate(all_cells_parts).reshape(-1).astype(np.int64)
                return (
                    points,
                    cells,
                    np.asarray(all_types, dtype=np.int32),
                    np.asarray(all_eletypes, dtype=object),
                )

        if {"node_coords", "elem_conn"}.issubset(names):
            try:
                points = _squeeze(np.asarray(src["node_coords"], dtype=np.float64))
                if points.ndim == 1:
                    dim = 3 if points.size % 3 == 0 else 2
                    points = points.reshape((-1, dim))
                conn = _squeeze(np.asarray(src["elem_conn"], dtype=np.int64))
                if conn.ndim == 1:
                    conn = conn.reshape(1, -1)
                labels = _squeeze(np.asarray(src["node_labels"], dtype=np.int64)) if "node_labels" in names else None
                if labels is not None and labels.size and conn.size and int(np.min(conn)) >= 1:
                    label_to_idx = {int(label): idx for idx, label in enumerate(labels.reshape(-1))}
                    conn = np.vectorize(lambda item: label_to_idx.get(int(item), int(item) - 1))(conn).astype(np.int64)
                ele_raw = src["ele_type"] if "ele_type" in names else ""
                ele_key = str(np.asarray(ele_raw).reshape(-1)[0]).upper() if np.asarray(ele_raw).size else ""
                vtk_type = _ABAQUS_TO_VTK.get(ele_key, 5 if conn.shape[1] == 3 else 9 if conn.shape[1] == 4 else 12)
                prefix = np.full((conn.shape[0], 1), conn.shape[1], dtype=np.int64)
                cells = np.hstack([prefix, conn]).reshape(-1).astype(np.int64)
                return (
                    points,
                    cells,
                    np.full(conn.shape[0], vtk_type, dtype=np.int32),
                    np.asarray([ele_key or f"N{conn.shape[1]}"] * conn.shape[0], dtype=object),
                )
            except Exception:
                return None, None, None, None

        return None, None, None, None

    # plain dict-style InpData
    if isinstance(inp, dict):
        points = inp.get("points")
        if points is None and {"node_coords", "elem_conn"}.issubset(inp.keys()):
            try:
                points = _squeeze(np.asarray(inp.get("node_coords"), dtype=np.float64))
                if points.ndim == 1:
                    dim = 3 if points.size % 3 == 0 else 2
                    points = points.reshape((-1, dim))
                conn = _squeeze(np.asarray(inp.get("elem_conn"), dtype=np.int64))
                if conn.ndim == 1:
                    conn = conn.reshape(1, -1)
                labels = _squeeze(np.asarray(inp.get("node_labels"), dtype=np.int64)) if inp.get("node_labels") is not None else None
                if labels is not None and labels.size and conn.size and int(np.min(conn)) >= 1:
                    label_to_idx = {int(label): idx for idx, label in enumerate(labels.reshape(-1))}
                    conn = np.vectorize(lambda item: label_to_idx.get(int(item), int(item) - 1))(conn).astype(np.int64)
                ele_raw = inp.get("ele_type", "")
                ele_arr = np.asarray(ele_raw)
                ele_key = str(ele_arr.reshape(-1)[0]).upper() if ele_arr.size else ""
                vtk_type = _ABAQUS_TO_VTK.get(ele_key, 5 if conn.shape[1] == 3 else 9 if conn.shape[1] == 4 else 12)
                prefix = np.full((conn.shape[0], 1), conn.shape[1], dtype=np.int64)
                cells = np.hstack([prefix, conn]).reshape(-1).astype(np.int64)
                return (
                    points,
                    cells,
                    np.full(conn.shape[0], vtk_type, dtype=np.int32),
                    np.asarray([ele_key or f"N{conn.shape[1]}"] * conn.shape[0], dtype=object),
                )
            except Exception:
                return None, None, None, None
        if points is None:
            return None, None, None, None
        points = _squeeze(np.asarray(points)).astype(np.float64)

        ctn = inp.get("cell_block_cell_types", [])
        etn = inp.get("cell_block_ele_types", ctn)
        cblocks = inp.get("cell_block_cells", [])

        if hasattr(ctn, "flat"):
            ctn = [str(np.asarray(x).flat[0]) for x in np.asarray(ctn).flat]
        if hasattr(etn, "flat"):
            etn = [str(np.asarray(x).flat[0]) for x in np.asarray(etn).flat]
        if hasattr(cblocks, "flat"):
            cblocks = [np.asarray(x, dtype=np.int64) for x in np.asarray(cblocks).flat]

        all_cells_parts: list[np.ndarray] = []
        all_types: list[int] = []
        all_eletypes: list[str] = []

        for idx, conn_raw in enumerate(cblocks):
            conn = _squeeze(np.asarray(conn_raw)).astype(np.int64)
            if conn.ndim < 2:
                continue
            n_elem, npe = conn.shape
            ele_key = str(etn[idx]).upper() if idx < len(etn) else ""
            vtk_type = _ABAQUS_TO_VTK.get(ele_key, 12)
            prefix = np.full((n_elem, 1), npe, dtype=np.int64)
            all_cells_parts.append(np.hstack([prefix, conn]))
            all_types.extend([vtk_type] * n_elem)
            all_eletypes.extend([ele_key] * n_elem)

        if not all_cells_parts:
            return None, None, None, None
        cells = np.concatenate(all_cells_parts).reshape(-1).astype(np.int64)
        return (
            points,
            cells,
            np.asarray(all_types, dtype=np.int32),
            np.asarray(all_eletypes, dtype=object),
        )

    return None, None, None, None


def _infer_component_count(arr: np.ndarray, vd: VizData, location: str) -> int:
    """Infer component count for gauss/node arrays stored in different layouts."""
    data = np.asarray(arr)
    if data.ndim >= 2:
        last = data.shape[-1]
        if last in (1, 2, 3, 6):
            return int(last)

    if location == "node" and vd.points is not None and data.ndim == 1:
        n_nodes = int(vd.points.shape[0])
        if n_nodes > 0 and data.size % n_nodes == 0:
            n_comp = data.size // n_nodes
            if 1 <= n_comp <= 6:
                return int(n_comp)

    if location == "gauss" and vd.n_gauss_per_elem is not None and data.ndim == 1:
        n_gauss_total = int(np.sum(vd.n_gauss_per_elem))
        if n_gauss_total > 0 and data.size % n_gauss_total == 0:
            n_comp = data.size // n_gauss_total
            if n_comp in (1, 3, 6):
                return int(n_comp)

    if location == "element" and vd.cell_types is not None and data.ndim == 1:
        n_elem = int(vd.cell_types.shape[0])
        if n_elem > 0 and data.size % n_elem == 0:
            n_comp = data.size // n_elem
            if 1 <= n_comp <= 6:
                return int(n_comp)

    return 1


def _reshape_nodal_field_if_flat(arr: np.ndarray, vd: VizData) -> np.ndarray:
    """Restore flat nodal DOF arrays to ``(..., n_node, n_comp)`` when possible."""
    data = np.asarray(arr, dtype=np.float64)
    if vd.points is None:
        return data
    n_nodes = int(vd.points.shape[0])
    if n_nodes <= 0:
        return data
    if data.ndim == 1 and data.size % n_nodes == 0:
        n_comp = data.size // n_nodes
        if 1 <= n_comp <= 6:
            return data.reshape(n_nodes, n_comp)
    if data.ndim == 2 and data.shape[1] % n_nodes == 0:
        n_comp = data.shape[1] // n_nodes
        if 1 <= n_comp <= 6:
            return data.reshape(data.shape[0], n_nodes, n_comp)
    return data


# Single-frame field discovery patterns
_SINGLE_FRAME_FIELDS: list[tuple[tuple[str, ...], str, str]] = [
    (("gauss_stress", "elastic_gauss_stress"), "应力", "gauss"),
    (("gauss_strain",), "应变", "gauss"),
    (("gauss_plastic_strain",), "塑性应变", "gauss"),
    (("gauss_peeq",), "PEEQ", "gauss"),
    (("solid_u_nodal", "u", "elastic_u"), "位移", "node"),
    (("solid_nforc_nodal", "internal_force"), "NFORC", "node"),
    (("reaction",), "反力", "node"),
    (("field_residual_stress_a0", "residual_stress_a0", "residual_stress"), "残余应力", "gauss"),
]

# Multi-frame field discovery patterns (first dim = n_frames)
_MULTI_FRAME_FIELDS: list[tuple[tuple[str, ...], str, str]] = [
    (("frame_gauss_stress",), "应力（历程）", "gauss"),
    (("frame_gauss_strain",), "应变（历程）", "gauss"),
    (("frame_gauss_plast_strain", "frame_gauss_plastic_strain"), "塑性应变（历程）", "gauss"),
    (("frame_gauss_peeq",), "PEEQ（历程）", "gauss"),
    (("frame_u",), "位移（历程）", "node"),
    (("frame_nforc", "frame_internal_force", "NFORC"), "NFORC（历程）", "node"),
    (("frame_force_error",), "自由节点平衡残差（历程）", "node"),
    (("frame_reaction",), "反力（历程）", "node"),
    (("field_residual_stress_cycle", "residual_stress_cycle"), "残余应力（历程）", "gauss"),
    (("field_total_stress_cycle", "total_stress_cycle"), "总应力（历程）", "gauss"),
    (("field_excess_stress_cycle", "excess_stress_cycle"), "超额应力（历程）", "gauss"),
]


def _known_field_descriptors() -> dict[str, tuple[str, str, bool]]:
    """Return known field descriptors: key -> (label, location, multiframe)."""
    out: dict[str, tuple[str, str, bool]] = {}
    for keys, label, loc in _SINGLE_FRAME_FIELDS:
        for key in keys:
            out[key] = (label, loc, False)
    for keys, label, loc in _MULTI_FRAME_FIELDS:
        for key in keys:
            out[key] = (label, loc, True)
    return out


_KNOWN_FIELD_DESCRIPTORS = _known_field_descriptors()

_HIDDEN_VISUAL_FIELDS = {"internal_force", "frame_internal_force"}


_FIELD_DISPLAY_METADATA: dict[str, dict[str, str]] = {
    "residual_stress_snapshot": {
        "symbol": r"\boldsymbol{\rho}",
        "formula": r"\boldsymbol{\rho}=\boldsymbol{\rho}(t_m),\quad t_m\approx T/2",
        "description": "Single converged residual-stress snapshot for RSDM-S Diff comparison",
    },
    "residual_stress_a0": {
        "symbol": r"\boldsymbol{\rho}_0",
        "formula": r"\boldsymbol{\rho}_0=\frac{1}{T}\int_0^T\boldsymbol{\rho}(t)\,dt",
    },
    "residual_stress_cycle": {
        "symbol": r"\boldsymbol{\rho}(t)",
        "formula": r"\boldsymbol{\rho}(t)=\boldsymbol{\sigma}(t)-\boldsymbol{\sigma}^{el}(t)",
    },
    "total_stress_cycle": {
        "symbol": r"\boldsymbol{\sigma}(t)",
        "formula": r"\boldsymbol{\sigma}(t)=\boldsymbol{\sigma}^{el}(t)+\boldsymbol{\rho}(t)",
    },
    "excess_stress_cycle": {
        "symbol": r"\boldsymbol{\sigma}^{cs}_{p}(t)",
        "formula": r"\boldsymbol{\sigma}^{cs}_{p}(t)=\operatorname{excess}\left(\boldsymbol{\sigma}(t),\sigma_y\right)",
    },
    "effective_excess_integral": {
        "symbol": r"\boldsymbol{\alpha}",
        "formula": r"\alpha_i=\int_0^T\sigma^{cs}_{p,i}(t)\,dt",
    },
    "alpha_norm_per_gp": {
        "symbol": r"\|\boldsymbol{\alpha}\|_2",
        "formula": r"\|\boldsymbol{\alpha}\|_2=\left(\sum_i\alpha_i^2\right)^{1/2}",
    },
    "max_abs_excess_component_per_gp": {
        "symbol": r"m_i",
        "formula": r"m_i=\max_{t\in[0,T]}\left|\sigma^{cs}_{p,i}(t)\right|",
    },
    "max_abs_excess_per_gp": {
        "symbol": r"m_{\max}",
        "formula": r"m_{\max}=\max_{t,i}\left|\sigma^{cs}_{p,i}(t)\right|",
    },
    "max_effective_excess_per_gp": {
        "symbol": r"\sigma^{cs}_{p,vm,\max}",
        "formula": r"\sigma^{cs}_{p,vm,\max}=\max_{t\in[0,T]}\sigma^{cs}_{p,vm}(t)",
    },
    "rsdms_XS_path": {
        "symbol": r"\sigma^{cs}_{p,vm,\mathrm{path}}",
        "formula": r"\sigma^{cs}_{p,vm,\mathrm{path}}=\max_{\mathrm{segment}}\sigma^{cs}_{p,vm}",
        "description": "Maximum von Mises effective excess stress on each RSDM-S loading path segment",
    },
    "rsdms_XS_path_SNEG": {
        "symbol": r"\sigma^{cs}_{p,vm,\mathrm{path}}^{SNEG}",
        "formula": r"\sigma^{cs}_{p,vm,\mathrm{path}}=\max_{\mathrm{segment}}\sigma^{cs}_{p,vm}",
        "description": "Maximum von Mises effective excess stress on each RSDM-S loading path segment, SNEG surface",
    },
    "rsdms_XS_path_SPOS": {
        "symbol": r"\sigma^{cs}_{p,vm,\mathrm{path}}^{SPOS}",
        "formula": r"\sigma^{cs}_{p,vm,\mathrm{path}}=\max_{\mathrm{segment}}\sigma^{cs}_{p,vm}",
        "description": "Maximum von Mises effective excess stress on each RSDM-S loading path segment, SPOS surface",
    },
    "rsdms_XS_vmmax": {
        "symbol": r"\sigma^{cs}_{p,vm,\max}",
        "formula": r"\sigma^{cs}_{p,vm,\max}=\max_{\mathrm{cycle}}\sigma^{cs}_{p,vm}",
        "description": "Maximum von Mises effective excess stress over the whole RSDM-S loading cycle",
    },
    "rsdm_XS_path": {
        "symbol": r"\sigma^{cs}_{p,vm,\mathrm{path}}",
        "formula": r"\sigma^{cs}_{p,vm,\mathrm{path}}=\max_{\mathrm{path}}\sigma^{cs}_{p,vm}(t)",
        "description": "Maximum von Mises effective excess stress on each RSDM loading path segment",
    },
    "classification_ratio_per_gp": {
        "symbol": r"r_g",
        "formula": r"r_g=\frac{\|\int_0^T\boldsymbol{\sigma}^{cs}_{p,g}(t)\,dt\|_2}{\max_t\sigma^{cs}_{p,vm,g}(t)}",
    },
    "gauss_state_map": {
        "symbol": r"s_g",
        "formula": r"s_g\in\{0,1,2\}\quad(\mathrm{shakedown},\mathrm{alternating},\mathrm{ratcheting})",
    },
    "elastic_gauss_stress": {
        "symbol": r"\boldsymbol{\sigma}^{el}",
        "formula": r"\boldsymbol{\sigma}^{el}=\mathbb{C}:\boldsymbol{\varepsilon}^{el}",
    },
    "elastic_u": {
        "symbol": r"\mathbf{u}^{el}",
        "formula": r"\mathbf{K}\mathbf{u}^{el}=\mathbf{f}",
    },
    "frame_gauss_stress": {
        "symbol": r"\boldsymbol{\sigma}(t)",
        "formula": r"\boldsymbol{\sigma}(t)=\mathbb{C}:\left(\boldsymbol{\varepsilon}(t)-\boldsymbol{\varepsilon}^p(t)\right)",
    },
    "frame_gauss_strain": {
        "symbol": r"\boldsymbol{\varepsilon}(t)",
        "formula": r"\boldsymbol{\varepsilon}(t)=\mathbf{B}\mathbf{u}(t)",
    },
    "frame_gauss_peeq": {
        "symbol": r"\bar{\varepsilon}^{p}(t)",
        "formula": r"\bar{\varepsilon}^{p}(t)=\int_0^t\sqrt{\frac{2}{3}\dot{\boldsymbol{\varepsilon}}^p:\dot{\boldsymbol{\varepsilon}}^p}\,d\tau",
    },
    "frame_gauss_plast_strain": {
        "symbol": r"\boldsymbol{\varepsilon}^{p}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{p}_{n+1}=\boldsymbol{\varepsilon}^{p}_{n}+\Delta\lambda\,\mathbf{n}",
    },
    "frame_u": {
        "symbol": r"\mathbf{u}(t)",
        "formula": r"\mathbf{u}(t)=\sum_i u_i(t)\mathbf{e}_i",
    },
    "frame_internal_force": {
        "symbol": r"\mathbf{f}_{int}(t)",
        "formula": r"\mathbf{f}_{int}(t)=\int_{\Omega}\mathbf{B}^{T}\boldsymbol{\sigma}(t)\,d\Omega",
    },
    "frame_nforc": {
        "symbol": r"\mathbf{NFORC}(t)",
        "formula": r"\mathbf{NFORC}(t)=\int_{\Omega_e}\mathbf{B}^{T}\boldsymbol{\sigma}(t)\,d\Omega",
    },
    "frame_force_error": {
        "symbol": r"\mathbf{r}_{free}(t)",
        "formula": r"\mathbf{r}_{free}(t)=\mathbf{f}_{int,free}(t)-\mathbf{f}_{ext,free}(t)",
    },
    "frame_reaction": {
        "symbol": r"\mathbf{R}(t)",
        "formula": r"\mathbf{R}(t)=\mathbf{f}_{int,bc}(t)-\mathbf{f}_{ext,bc}(t)",
    },
    "shakedown_residual_stress": {
        "symbol": r"\boldsymbol{\rho}",
        "formula": r"\boldsymbol{\rho}=\boldsymbol{\sigma}^{tot}-\alpha\boldsymbol{\sigma}^{E}",
        "description": "Self-equilibrated residual stress from the shakedown optimization",
    },
    "shakedown_total_stress": {
        "symbol": r"\boldsymbol{\sigma}^{tot}",
        "formula": r"\boldsymbol{\sigma}^{tot}_v=\boldsymbol{\rho}+\alpha\boldsymbol{\sigma}^{E}_v",
        "description": "Total stress at each load-domain vertex",
    },
    "shakedown_inequality_violation": {
        "symbol": r"C_{\mathrm{ineq}}",
        "formula": r"C_{\mathrm{ineq}}=\max(\sigma_{\mathrm{eq}}/\sigma_y-1,0)",
        "description": "Yield inequality violation; zero means the Gauss point satisfies the yield constraint",
    },
    "shakedown_inequality_multiplier": {
        "symbol": r"\lambda_y",
        "formula": r"\lambda_y\ge 0,\quad \lambda_y\,g_y(\boldsymbol{\sigma}^{tot})=0",
        "description": "Dual multiplier of the yield inequality; larger values indicate active plastic-yield constraints",
    },
    "shakedown_equality_violation": {
        "symbol": r"C_{\mathrm{eq}}",
        "formula": r"C_{\mathrm{eq}}=\mathbf{C}\boldsymbol{\rho}",
        "description": "Residual-stress self-equilibrium residual on free DOFs",
    },
}


def _field_family(key: str, label: str = "") -> str:
    key_lower = str(key).lower()
    text = f"{key} {label}".lower()
    if "stress" in text:
        return "stress"
    if "strain" in text or "peeq" in text:
        return "strain"
    if key_lower in {"u", "frame_u", "elastic_u", "solid_u_nodal"} or "displacement" in text:
        return "displacement"
    if "nforc" in text or "reaction" in text or "internal_force" in text or "force" in text:
        return "force"
    return "other"


def _tensor_kind(family: str, n_components: int, key: str) -> str:
    if "peeq" in key.lower():
        return "scalar"
    if family in {"displacement", "force"}:
        return "vector"
    if n_components <= 1:
        return "scalar"
    if family == "stress" and n_components in (3, 6):
        return "stress_voigt"
    if family == "strain" and n_components in (3, 6):
        return "strain_voigt"
    return "components"


def _infer_components_from_shape(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    last = int(shape[-1])
    if last in (1, 2, 3, 6):
        return last
    return 1


def _is_numeric_mat_class(mat_class: str) -> bool:
    return str(mat_class).lower() in {
        "double",
        "single",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "logical",
    }


def _is_visualizable_shape(shape: Sequence[int]) -> bool:
    dims = [int(v) for v in shape if int(v) > 1]
    if not dims:
        return False
    if len(dims) > 4:
        return False
    return True


def _infer_field_context_from_whos(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict:
    """Infer node/element/gauss counts from MAT variable shapes."""
    nodes: set[int] = set()
    elems: set[int] = set()
    gauss_totals: set[int] = set()

    for name, shape_raw, mat_class in entries:
        if not _is_numeric_mat_class(mat_class):
            continue
        shape = tuple(int(v) for v in shape_raw if int(v) > 0)
        if not shape:
            continue
        lowered = name.lower()
        if "_time_" in lowered or lowered.endswith("_time"):
            continue

        if name == "viz_points" and len(shape) >= 2:
            nodes.add(shape[0])

        if lowered in {"solid_u_nodal", "solid_nforc_nodal"} and len(shape) >= 2:
            nodes.add(shape[0])
        elif lowered in {"frame_u", "frame_nforc", "frame_internal_force", "frame_force_error"}:
            if len(shape) >= 3:
                nodes.add(shape[1])
        elif lowered in {"u", "elastic_u", "internal_force", "reaction"}:
            if len(shape) >= 2 and shape[-1] in (1, 2, 3, 6):
                nodes.add(shape[0])

        if lowered == "gauss_coords":
            if len(shape) == 3 and shape[-1] in (2, 3):
                elems.add(shape[0])
                gauss_totals.add(shape[0] * shape[1])
            elif len(shape) == 2 and shape[-1] in (2, 3):
                gauss_totals.add(shape[0])
        elif lowered == "gauss_vols":
            dims = [v for v in shape if v > 1]
            if dims:
                gauss_totals.add(dims[0])

        if "gauss" in lowered or "peeq" in lowered or "residual_stress" in lowered or "total_stress" in lowered:
            if len(shape) == 4 and shape[-1] in (1, 2, 3, 6):
                elems.add(shape[1])
                gauss_totals.add(shape[1] * shape[2])
            elif len(shape) == 3:
                if shape[-1] in (1, 2, 3, 6):
                    # Could be (n_elem, n_gp, n_comp) or (n_frame, n_gauss, n_comp).
                    if lowered.startswith("frame_") or lowered.endswith("_cycle"):
                        gauss_totals.add(shape[1])
                    else:
                        elems.add(shape[0])
                        gauss_totals.add(shape[0] * shape[1])
                elif lowered.startswith("frame_") or lowered.endswith("_cycle"):
                    elems.add(shape[1])
                    gauss_totals.add(shape[1] * shape[2])
            elif len(shape) == 2:
                if lowered.startswith("frame_") or lowered.endswith("_cycle"):
                    gauss_totals.add(shape[1])
                elif shape[-1] in (1, 2, 3, 4, 6):
                    gauss_totals.add(shape[0])
                else:
                    elems.add(shape[0])
                    gauss_totals.add(shape[0] * shape[1])
            elif len(shape) == 1:
                gauss_totals.add(shape[0])

    return {"nodes": nodes, "elements": elems, "gauss": gauss_totals}


def _augment_field_context_from_inpdata(mat_path: Path, context: dict, entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict:
    """Add mesh counts from ``InpData`` when whosmat lacks explicit aliases."""
    if not any(name == "InpData" for name, _, _ in entries):
        return context
    if context.get("nodes"):
        return context

    import scipy.io as sio

    try:
        raw = sio.loadmat(str(mat_path), variable_names=["InpData"])
    except Exception:
        return context

    inp = raw.get("InpData")
    points = _struct_get(inp, "points")
    points_arr = _as_float_array(points)
    if points_arr is None:
        return context
    if points_arr.ndim >= 2 and points_arr.shape[0] > 0:
        context.setdefault("nodes", set()).add(int(points_arr.shape[0]))
    return context


def _shape_has_count(shape: Sequence[int], counts: set[int]) -> bool:
    return any(int(v) in counts for v in shape)


def _match_field_layout(
    name: str,
    shape_raw: Sequence[int],
    descriptor: Optional[tuple[str, str, bool]],
    context: dict,
) -> Optional[dict]:
    """Return visual field metadata if *shape* matches mesh field dimensions."""
    shape = tuple(int(v) for v in shape_raw if int(v) > 0)
    if not shape:
        return None

    lowered = name.lower()
    if "_time_" in lowered or lowered.endswith("_time"):
        return None
    field_like = (
        descriptor is not None
        or any(token in lowered for token in (
            "stress", "strain", "peeq", "displacement", "nforc",
            "reaction", "internal_force", "force", "yield_ratio", "violation",
        ))
        or lowered in {"u", "elastic_u"}
    )
    if not field_like:
        return None

    if descriptor is not None:
        _, location, is_frames = descriptor
    else:
        family = _field_family(name)
        is_frames = lowered.startswith("frame_") or lowered.endswith("_cycle")
        if any(token in lowered for token in ("gauss", "stress", "strain", "peeq", "yield_ratio", "violation")):
            location = "gauss"
        elif family in {"displacement", "force"} or "nodal" in lowered or lowered in {"u", "elastic_u"}:
            location = "node"
        else:
            location = "unknown"

    nodes = context.get("nodes", set())
    elems = context.get("elements", set())
    gauss = context.get("gauss", set())
    n_components = _infer_components_from_shape(shape)

    if location == "node":
        if len(shape) >= 3 and shape[1] in nodes and shape[-1] in (1, 2, 3, 6):
            return {"location": "node", "frames": True, "n_components": int(shape[-1])}
        if len(shape) >= 2 and shape[0] in nodes and shape[-1] in (1, 2, 3, 6):
            return {"location": "node", "frames": False, "n_components": int(shape[-1])}
        # Flat nodal DOF layouts are accepted only for known displacement/force fields.
        flat_ok = descriptor is not None or lowered in {"u", "elastic_u", "internal_force", "reaction"}
        if flat_ok and nodes:
            if len(shape) == 2 and shape[0] > 1:
                for n_node in nodes:
                    if shape[1] % n_node == 0 and 1 <= shape[1] // n_node <= 6:
                        return {"location": "node", "frames": True, "n_components": int(shape[1] // n_node)}
            total = int(np.prod([v for v in shape if v > 1]))
            for n_node in nodes:
                if total % n_node == 0 and 1 <= total // n_node <= 6:
                    return {"location": "node", "frames": bool(is_frames), "n_components": int(total // n_node)}
        return None

    if location == "gauss":
        if len(shape) == 4 and shape[1] in elems and shape[1] * shape[2] in gauss and shape[-1] in (1, 2, 3, 4, 6):
            return {"location": "gauss", "frames": True, "n_components": int(shape[-1])}
        if len(shape) == 3:
            if shape[1] in gauss and shape[-1] in (1, 2, 3, 4, 6):
                return {"location": "gauss", "frames": True, "n_components": int(shape[-1])}
            if shape[0] in elems and shape[0] * shape[1] in gauss and shape[-1] in (1, 2, 3, 4, 6):
                return {"location": "gauss", "frames": False, "n_components": int(shape[-1])}
            if shape[1] in elems and shape[1] * shape[2] in gauss:
                return {"location": "gauss", "frames": True, "n_components": 1}
        if len(shape) == 2:
            if shape[1] in gauss and (is_frames or lowered.startswith("frame_")):
                return {"location": "gauss", "frames": True, "n_components": 1}
            if shape[0] in gauss and shape[-1] in (1, 2, 3, 4, 6):
                return {"location": "gauss", "frames": False, "n_components": int(shape[-1])}
            if shape[0] in elems and shape[0] * shape[1] in gauss:
                return {"location": "gauss", "frames": False, "n_components": 1}
        if len(shape) == 1 and shape[0] in gauss:
            return {"location": "gauss", "frames": False, "n_components": 1}
        return None

    if _shape_has_count(shape, elems):
        if len(shape) == 2 and shape[0] in elems and shape[-1] in (1, 2, 3, 4, 6):
            return {"location": "element", "frames": False, "n_components": int(shape[-1])}
        if len(shape) == 1 and shape[0] in elems:
            return {"location": "element", "frames": False, "n_components": 1}

    return None


def _field_sort_key(item: dict) -> tuple:
    family_order = {"stress": 0, "strain": 1, "displacement": 2, "force": 3, "other": 4}
    loc_order = {"node": 0, "element": 1, "gauss": 2, "unknown": 3}
    return (
        0 if item.get("default_selected") else 1,
        family_order.get(str(item.get("family")), 9),
        loc_order.get(str(item.get("location")), 9),
        str(item.get("key", "")).lower(),
    )


def _manifest_default_selected(key: str, raw_field: Mapping[str, Any], available: bool = True) -> bool:
    """Return the reader-side default selection for manifest fields.

    Older shakedown MAT files may already carry a manifest where the inequality
    multiplier was marked optional even though the expensive dual reconstruction
    result is present.  Treat the field as default-visible once it exists.
    """
    if str(key) == "shakedown_inequality_multiplier" and available:
        return True
    if str(key) == "rsdms_XS_vmmax" and available:
        return True
    return bool(raw_field.get("default_selected", False))


def _entry_map(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict[str, tuple[tuple[int, ...], str]]:
    return {str(name): (tuple(int(v) for v in shape), str(mat_class)) for name, shape, mat_class in entries}


def _load_nested_schema_shapes(mat_path: Path) -> dict[str, tuple[int, ...]]:
    import scipy.io as sio

    try:
        raw = sio.loadmat(
            str(mat_path),
            squeeze_me=False,
            struct_as_record=False,
            variable_names=[FRAME_OUTPUTS_KEY, SHELL_PARAM_KEY, SOLID_PARAM_KEY],
        )
    except Exception:
        return {}
    shapes: dict[str, tuple[int, ...]] = {}
    if FRAME_OUTPUTS_KEY in raw:
        shapes.update(nested_frame_output_shapes(raw.get(FRAME_OUTPUTS_KEY)))
    if SHELL_PARAM_KEY in raw:
        shapes.update(nested_struct_shapes(raw.get(SHELL_PARAM_KEY)))
    if SOLID_PARAM_KEY in raw:
        shapes.update(nested_struct_shapes(raw.get(SOLID_PARAM_KEY)))
    return shapes


def _entries_with_nested_schema_shapes(
    entries: Sequence[tuple[str, tuple[int, ...], str]],
    nested_shapes: Mapping[str, tuple[int, ...]],
) -> list[tuple[str, tuple[int, ...], str]]:
    out = [(str(name), tuple(int(v) for v in shape), str(mat_class)) for name, shape, mat_class in entries]
    present = {name for name, _shape, _mat_class in out}
    for key, shape in nested_shapes.items():
        if key not in present:
            out.append((str(key), tuple(int(v) for v in shape), "nested"))
    return out


def _manifest_source_keys(field: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for name in ("key", "source_key"):
        value = field.get(name)
        if value is not None and str(value).strip():
            keys.append(str(value))
    for value in field.get("source_keys") or []:
        if value is not None and str(value).strip():
            keys.append(str(value))
    return keys


def _manifest_field_shape(field: Mapping[str, Any], entries_by_name: Mapping[str, tuple[tuple[int, ...], str]]) -> tuple[list[int], str]:
    explicit = field.get("shape")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        try:
            return [int(v) for v in explicit], str(field.get("mat_class", "declared"))
        except Exception:
            pass
    for key in _manifest_source_keys(field):
        if key in entries_by_name:
            shape, mat_class = entries_by_name[key]
            return [int(v) for v in shape], str(mat_class)
    return [], str(field.get("mat_class", "derived"))


def _fields_from_manifest(
    manifest: Mapping[str, Any],
    entries: Sequence[tuple[str, tuple[int, ...], str]],
) -> list[dict]:
    """Convert ``VizManifestJSON`` into inspect-field rows.

    A field is listed only when its declared key or at least one declared source
    key is present in the MAT file.  This keeps visualization scoped to fields
    that the producing module explicitly exposed.
    """
    entries_by_name = _entry_map(entries)
    fields: list[dict] = []
    for raw_field in manifest.get("fields") or []:
        if not isinstance(raw_field, Mapping):
            continue
        key = str(raw_field.get("key", "")).strip()
        if not key:
            continue
        if key in _HIDDEN_VISUAL_FIELDS:
            continue
        source_keys = _manifest_source_keys(raw_field)
        available = any(src in entries_by_name for src in source_keys)
        if not available and key not in _SHAKEDOWN_FIELD_SPECS:
            continue
        shape, mat_class = _manifest_field_shape(raw_field, entries_by_name)
        location = str(raw_field.get("location", "unknown"))
        family = str(raw_field.get("family", _field_family(key, str(raw_field.get("label", "")))))
        n_components = int(raw_field.get("n_components") or _infer_components_from_shape(shape) or 1)
        frames = str(raw_field.get("frames", "single"))
        display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
        fields.append({
            "key": key,
            "label": str(raw_field.get("label", key)),
            "shape": shape,
            "mat_class": mat_class,
            "location": location,
            "frames": frames,
            "frame_axis": str(raw_field.get("frame_axis", "")),
            "n_components": int(n_components),
            "family": family,
            "tensor_kind": str(raw_field.get("tensor_kind", _tensor_kind(family, n_components, key))),
            "default_selected": _manifest_default_selected(key, raw_field, available),
            "visualizable": True,
            "source_key": str(raw_field.get("source_key", "")),
            "source_keys": [str(item) for item in raw_field.get("source_keys") or []],
            "scalar_options": [str(item) for item in raw_field.get("scalar_options") or []],
            "symbol": str(raw_field.get("symbol") or display_meta.get("symbol", "")),
            "formula": str(raw_field.get("formula") or display_meta.get("formula", "")),
            "description": str(raw_field.get("description") or display_meta.get("description", "")),
        })
    fields.sort(key=_field_sort_key)
    return fields


def _apply_manifest_field_info(
    vd: VizData,
    manifest: Mapping[str, Any] | None,
    selected_fields: Optional[set[str]],
) -> None:
    if manifest is None:
        return
    for raw_field in manifest.get("fields") or []:
        if not isinstance(raw_field, Mapping):
            continue
        key = str(raw_field.get("key", "")).strip()
        if not key or key not in vd.field_info:
            continue
        if selected_fields is not None and key not in selected_fields:
            continue
        fi = vd.field_info[key]
        display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
        n_components = raw_field.get("n_components")
        try:
            n_comp = int(n_components) if n_components is not None else fi.n_components
        except Exception:
            n_comp = fi.n_components
        n_frames = fi.n_frames
        if str(raw_field.get("frames", "")).strip().lower() == "frames":
            arr = vd.fields.get(key)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                n_frames = int(arr.shape[0])
        vd.field_info[key] = FieldInfo(
            key=key,
            label=str(raw_field.get("label", fi.label)),
            n_components=n_comp,
            location=str(raw_field.get("location", fi.location)),
            n_frames=n_frames,
            symbol=str(raw_field.get("symbol") or fi.symbol or display_meta.get("symbol", "")),
            formula=str(raw_field.get("formula") or fi.formula or display_meta.get("formula", "")),
            description=str(raw_field.get("description") or fi.description or display_meta.get("description", "")),
        )


def _incremental_shell_display_spec(key: str) -> dict[str, str]:
    try:
        from jaxmech.modules.inc_analysis.plastic.visualize_mat import (
            shell_incremental_field_display_spec,
        )

        return shell_incremental_field_display_spec(key)
    except Exception:
        return {}


def _normalize_manifest_for_reader(
    manifest: Mapping[str, Any] | None,
    available_keys: Optional[Sequence[str]] = None,
) -> dict[str, Any] | None:
    """Apply reader-side migrations for manifests written by older local builds."""
    if manifest is None:
        return None
    out = dict(manifest)
    analysis_type = str(out.get("analysis_type", "")).strip().lower()
    family = str(out.get("family", "")).strip().lower()
    fields: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    available = {str(key) for key in (available_keys or [])}
    for raw_field in manifest.get("fields") or []:
        if not isinstance(raw_field, Mapping):
            continue
        field = dict(raw_field)
        key = str(field.get("key", "")).strip()
        if key in _HIDDEN_VISUAL_FIELDS:
            continue
        if analysis_type == "shakedown" and key in {"shakedown_residual_stress", "shakedown_equality_violation"}:
            field["frames"] = "single"
            field.pop("frame_axis", None)
        if analysis_type == "shakedown" and key == "shakedown_inequality_violation":
            field["label"] = "Shakedown inequality violation"
        if key == "shakedown_total_stress":
            source_keys = list(field.get("source_keys") or [])
            for source in ("residual_stress", "gauss_stress_cases", "sigma_E", "vertex_load_matrix", "objective_value"):
                if source not in source_keys:
                    source_keys.append(source)
            field["source_keys"] = source_keys
        if key == "shakedown_equality_violation":
            source_keys = list(field.get("source_keys") or [])
            for source in ("equilibrium_residual", "C_sparse", "residual_stress", "free_dofs"):
                if source not in source_keys:
                    source_keys.append(source)
            field["source_keys"] = source_keys
        if key in _SHAKEDOWN_FIELD_SPECS:
            spec = _SHAKEDOWN_FIELD_SPECS[key]
            for meta_name in ("symbol", "formula", "description"):
                if not field.get(meta_name):
                    field[meta_name] = spec.get(meta_name, "")
        if analysis_type == "nonlinear_static" and family == "shell":
            spec = _incremental_shell_display_spec(key)
            if spec:
                if spec.get("label"):
                    field["label"] = spec["label"]
                for meta_name in ("symbol", "formula", "description"):
                    if not field.get(meta_name):
                        field[meta_name] = spec.get(meta_name, "")
        seen_keys.add(key)
        fields.append(field)
    if analysis_type == "nonlinear_static" and family == "shell" and available:
        for key in (
            "shell_layer_plastic_strain",
            "frame_shell_layer_plastic_strain",
            "frame_shell_layer_plastic_strain_SNEG",
            "frame_shell_layer_plastic_strain_SPOS",
        ):
            if key in seen_keys or key not in available:
                continue
            spec = _incremental_shell_display_spec(key)
            fields.append({
                "key": key,
                "label": spec.get("label", key),
                "family": "strain",
                "location": "gauss",
                "frames": "frames" if key.startswith("frame_") else "single",
                "frame_axis": "increment" if key.startswith("frame_") else "",
                "tensor_kind": "strain_voigt",
                "n_components": 4,
                "default_selected": False,
                "scalar_options": ["components", "magnitude"],
                "symbol": spec.get("symbol", ""),
                "formula": spec.get("formula", ""),
                "description": spec.get("description", ""),
            })
            seen_keys.add(key)
    if analysis_type == "shakedown" and any(
        str(field.get("key", "")) in {"shakedown_SF_res", "shakedown_SM_res", "shakedown_SF_tot", "shakedown_SM_tot"}
        for field in fields
    ):
        fields = [
            field for field in fields
            if str(field.get("key", "")) not in {"shakedown_residual_stress", "shakedown_total_stress"}
        ]
        seen_keys.difference_update({"shakedown_residual_stress", "shakedown_total_stress"})
    if (
        analysis_type == "shakedown"
        and "shakedown_inequality_multiplier" not in seen_keys
        and "dual_variables_ineq" in available
    ):
        spec = _SHAKEDOWN_FIELD_SPECS["shakedown_inequality_multiplier"]
        fields.append({
            "key": "shakedown_inequality_multiplier",
            "label": spec["label"],
            "family": spec["family"],
            "location": "gauss",
            "tensor_kind": spec["tensor_kind"],
            "n_components": 1,
            "frames": "frames",
            "frame_axis": "vertex",
            "default_selected": bool(spec["default_selected"]),
            "source_key": "dual_variables_ineq",
            "scalar_options": ["scalar"],
            "symbol": spec.get("symbol", ""),
            "formula": spec.get("formula", ""),
            "description": spec.get("description", ""),
        })
    if (
        analysis_type == "rsdm_shakedown"
        and "rsdms_XS_path" not in seen_keys
        and family != "shell"
        and "shell_layer_stress_cases" not in available
        and ("ResultsSet" in available or {"yield_ratio", "vertex_load_matrix"} <= available)
    ):
        display_meta = _FIELD_DISPLAY_METADATA["rsdms_XS_path"]
        fields.append({
            "key": "rsdms_XS_path",
            "label": "RSDM-S path effective excess stress",
            "family": "other",
            "location": "gauss",
            "tensor_kind": "scalar",
            "n_components": 1,
            "frames": "frames",
            "frame_axis": "path",
            "default_selected": False,
            "source_key": "ResultsSet",
            "source_keys": ["ResultsSet", "yield_ratio", "vertex_load_matrix"],
            "scalar_options": ["scalar"],
            "symbol": display_meta.get("symbol", ""),
            "formula": display_meta.get("formula", ""),
            "description": display_meta.get("description", ""),
        })
    if (
        analysis_type == "rsdm_shakedown"
        and (family == "shell" or "shell_layer_stress_cases" in available)
        and ("ResultsSet" in available or {"yield_ratio", "vertex_load_matrix"} <= available)
    ):
        for key, label in (
            ("rsdms_XS_path_SNEG", "RSDM-S XS_path (inner/SNEG)"),
            ("rsdms_XS_path_SPOS", "RSDM-S XS_path (outer/SPOS)"),
        ):
            if key in seen_keys:
                continue
            display_meta = _FIELD_DISPLAY_METADATA[key]
            fields.append({
                "key": key,
                "label": label,
                "family": "other",
                "location": "gauss",
                "tensor_kind": "scalar",
                "n_components": 1,
                "frames": "frames",
                "frame_axis": "path",
                "default_selected": False,
                "source_key": "ResultsSet",
                "source_keys": ["ResultsSet", "yield_ratio", "vertex_load_matrix", "shell_layer_stress_cases"],
                "scalar_options": ["scalar"],
                "symbol": display_meta.get("symbol", ""),
                "formula": display_meta.get("formula", ""),
                "description": display_meta.get("description", ""),
            })
    if (
        (analysis_type in {"rsdm", ""} or analysis_type.startswith("rsdm_"))
        and "rsdm_XS_path" not in seen_keys
        and ({"rsdm_XS_cyc"} & available or {"excess_stress_cycle", "field_excess_stress_cycle"} & available)
    ):
        display_meta = _FIELD_DISPLAY_METADATA["rsdm_XS_path"]
        fields.append({
            "key": "rsdm_XS_path",
            "label": "XS_path",
            "family": "other",
            "location": "gauss",
            "tensor_kind": "scalar",
            "n_components": 1,
            "frames": "frames",
            "frame_axis": "path",
            "default_selected": False,
            "source_key": "rsdm_XS_cyc",
            "source_keys": ["rsdm_XS_cyc", "excess_stress_cycle", "field_excess_stress_cycle"],
            "scalar_options": ["scalar"],
            "symbol": display_meta.get("symbol", ""),
            "formula": display_meta.get("formula", ""),
            "description": display_meta.get("description", ""),
        })
    out["fields"] = fields
    return out


def _entry_names(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> set[str]:
    return {str(name) for name, _, _ in entries}


def _is_rsdm_shakedown_manifest(manifest: Mapping[str, Any] | None, entries: Sequence[tuple[str, tuple[int, ...], str]]) -> bool:
    if manifest is None:
        return False
    names = _entry_names(entries)
    if {"RSDMShakedownTrace", "RSDMShakedownInput"} & names:
        return True
    return str(manifest.get("analysis_type", "")).strip().lower() == "rsdm_shakedown"


def _rebuild_rsdm_shakedown_manifest_if_needed(
    mat_path: Path,
    manifest: Mapping[str, Any] | None,
    entries: Sequence[tuple[str, tuple[int, ...], str]],
) -> dict[str, Any] | None:
    if not _is_rsdm_shakedown_manifest(manifest, entries):
        return dict(manifest) if isinstance(manifest, Mapping) else None
    producer = str((manifest or {}).get("producer_module", "")).strip()
    analysis_type = str((manifest or {}).get("analysis_type", "")).strip().lower()
    has_multiplier = any(
        isinstance(field, Mapping) and str(field.get("key", "")) == "shakedown_inequality_multiplier"
        for field in (manifest or {}).get("fields", [])
    )
    manifest_meta = (manifest or {}).get("metadata", {})
    if not isinstance(manifest_meta, Mapping):
        manifest_meta = {}
    is_rsdms_summary_manifest = str(manifest_meta.get("result_layout", "")).strip() == "rsdm_shakedown_summary"
    if (
        producer == "jaxmech.modules.direct_methods.rsdm"
        and analysis_type == "rsdm_shakedown"
        and not has_multiplier
        and is_rsdms_summary_manifest
    ):
        return dict(manifest) if isinstance(manifest, Mapping) else None
    try:
        from jaxmech.modules.direct_methods.rsdm.visualize_mat import build_viz_manifest_from_mat

        rebuilt = build_viz_manifest_from_mat(mat_path)
    except Exception:
        rebuilt = dict(manifest) if isinstance(manifest, Mapping) else None
        if rebuilt is not None:
            rebuilt["producer_module"] = "jaxmech.modules.direct_methods.rsdm"
            rebuilt["analysis_type"] = "rsdm_shakedown"
            rebuilt["fields"] = [
                field for field in rebuilt.get("fields", [])
                if not (isinstance(field, Mapping) and str(field.get("key", "")) == "shakedown_inequality_multiplier")
            ]
    if rebuilt is not None:
        try:
            write_viz_manifest_to_mat(mat_path, rebuilt)
        except Exception:
            pass
    return rebuilt


def _raw_is_rsdm_shakedown(raw: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> bool:
    if "RSDMShakedownTrace" in raw or "RSDMShakedownInput" in raw:
        return True
    if manifest is not None and str(manifest.get("analysis_type", "")).strip().lower() == "rsdm_shakedown":
        return True
    metadata = _extract_metadata(dict(raw))
    return str(metadata.get("analysis_type", "")).strip().lower() == "rsdm_shakedown"


def _filter_rsdm_shakedown_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    out = dict(manifest)
    out["producer_module"] = "jaxmech.modules.direct_methods.rsdm"
    out["analysis_type"] = "rsdm_shakedown"
    out["fields"] = [
        field for field in out.get("fields", [])
        if not (isinstance(field, Mapping) and str(field.get("key", "")) == "shakedown_inequality_multiplier")
    ]
    return out


def _build_legacy_manifest(mat_path: Path, entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict | None:
    """Build and persist a module manifest for a legacy MAT if possible."""
    names = _entry_names(entries)
    builders = []
    if {"DirectCyclicResult", "DirectCyclicInput"} & names or any(name.startswith("frame_") for name in names):
        builders.append("jaxmech.modules.direct_methods.dca.visualize_mat")
    if {"PlasticResult", "PlasticStateLayout", "plastic_state_layout", "frame_increment_index", "gauss_plastic_strain"} & names:
        builders.insert(0, "jaxmech.modules.inc_analysis.plastic.visualize_mat")
    if {"residual_stress_cycle", "total_stress_cycle", "excess_stress_cycle"} & names:
        builders.insert(0, "jaxmech.modules.direct_methods.rsdm.visualize_mat")
    if {"ElasticInputSet", "ResultsSet", "yield_ratio", "residual_stress", "sigma_E"} & names:
        builders.insert(0, "jaxmech.modules.shakedown.visualize_mat")
    if {"gauss_stress", "gauss_strain", "solid_u_nodal", "shell_u_nodal", "u", "internal_force"} & names:
        builders.append("jaxmech.modules.inc_analysis.visualize_mat")

    seen: set[str] = set()
    for module_name in builders:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = __import__(module_name, fromlist=["build_viz_manifest_from_mat"])
            manifest = module.build_viz_manifest_from_mat(mat_path)
        except Exception:
            continue
        if isinstance(manifest, dict) and manifest.get("fields"):
            try:
                write_viz_manifest_to_mat(mat_path, manifest)
            except Exception:
                pass
            return manifest
    return None


_SHAKEDOWN_FIELD_SPECS: dict[str, dict] = {
    "shakedown_total_stress": {
        "label": "Shakedown total stress",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        **_FIELD_DISPLAY_METADATA["shakedown_total_stress"],
    },
    "shakedown_residual_stress": {
        "label": "Shakedown residual stress",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        **_FIELD_DISPLAY_METADATA["shakedown_residual_stress"],
    },
    "shakedown_inequality_violation": {
        "label": "Shakedown inequality violation",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        **_FIELD_DISPLAY_METADATA["shakedown_inequality_violation"],
    },
    "shakedown_inequality_multiplier": {
        "label": "Yield inequality Lagrange multiplier",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        **_FIELD_DISPLAY_METADATA["shakedown_inequality_multiplier"],
    },
    "shakedown_equality_violation": {
        "label": "Residual self-equilibrium violation",
        "family": "other",
        "tensor_kind": "vector",
        "default_selected": True,
        **_FIELD_DISPLAY_METADATA["shakedown_equality_violation"],
    },
    "max_effective_excess_per_gp": {
        "label": "Max effective excess stress",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": False,
        **_FIELD_DISPLAY_METADATA["max_effective_excess_per_gp"],
    },
    "shakedown_S_res_SNEG": {
        "label": "Shakedown S residual (inner/SNEG)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\rho}^{S}",
        "formula": r"\boldsymbol{\rho}^{S}_{k}=\boldsymbol{\rho}^{S}(z_k)",
        "description": "Layer Cauchy residual stress on the selected shell surface",
    },
    "shakedown_S_res_SPOS": {
        "label": "Shakedown S residual (outer/SPOS)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\rho}^{S}",
        "formula": r"\boldsymbol{\rho}^{S}_{k}=\boldsymbol{\rho}^{S}(z_k)",
        "description": "Layer Cauchy residual stress on the selected shell surface",
    },
    "shakedown_S_tot_SNEG": {
        "label": "Shakedown S total (inner/SNEG)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\sigma}^{S}_{tot}",
        "formula": r"\boldsymbol{\sigma}^{S}_{v,k}=\boldsymbol{\rho}^{S}_{k}+\alpha\boldsymbol{\sigma}^{E,S}_{v,k}",
        "description": "Layer Cauchy total stress reconstructed at each load-domain vertex",
    },
    "shakedown_S_tot_SPOS": {
        "label": "Shakedown S total (outer/SPOS)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\sigma}^{S}_{tot}",
        "formula": r"\boldsymbol{\sigma}^{S}_{v,k}=\boldsymbol{\rho}^{S}_{k}+\alpha\boldsymbol{\sigma}^{E,S}_{v,k}",
        "description": "Layer Cauchy total stress reconstructed at each load-domain vertex",
    },
    "shakedown_CInEQ_SNEG": {
        "label": "Shakedown CInEQ (inner/SNEG)",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        "symbol": r"C_{\mathrm{ineq}}",
        "formula": r"C_{\mathrm{ineq},v,k}=\max(\sigma_{vm}(\boldsymbol{\sigma}^{S}_{v,k})/\sigma_y-1,0)",
        "description": "Yield inequality violation on the selected shell surface",
    },
    "shakedown_CInEQ_SPOS": {
        "label": "Shakedown CInEQ (outer/SPOS)",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        "symbol": r"C_{\mathrm{ineq}}",
        "formula": r"C_{\mathrm{ineq},v,k}=\max(\sigma_{vm}(\boldsymbol{\sigma}^{S}_{v,k})/\sigma_y-1,0)",
        "description": "Yield inequality violation on the selected shell surface",
    },
    "shakedown_SF_res": {
        "label": "Shakedown SF residual",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\mathbf{SF}_{\rho}",
        "formula": r"\mathbf{SF}_{\rho}",
        "description": "Shell generalized membrane force residual",
    },
    "shakedown_SM_res": {
        "label": "Shakedown SM residual",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\mathbf{SM}_{\rho}",
        "formula": r"\mathbf{SM}_{\rho}",
        "description": "Shell generalized bending moment residual",
    },
    "shakedown_SF_tot": {
        "label": "Shakedown SF total",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\mathbf{SF}_{tot}",
        "formula": r"\mathbf{SF}_{v}=\mathbf{SF}_{\rho}+\alpha\mathbf{SF}^{E}_{v}",
        "description": "Shell total generalized membrane force at each load-domain vertex",
    },
    "shakedown_SM_tot": {
        "label": "Shakedown SM total",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\mathbf{SM}_{tot}",
        "formula": r"\mathbf{SM}_{v}=\mathbf{SM}_{\rho}+\alpha\mathbf{SM}^{E}_{v}",
        "description": "Shell total generalized bending moment at each load-domain vertex",
    },
    "shakedown_Phi": {
        "label": "Shakedown Phi (Ilyushin)",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        "symbol": r"\varphi_I",
        "formula": r"\varphi_I=\varphi_N+\varphi_M+2c|\varphi_{NM}|",
        "description": "Ilyushin yield function value; values near 1 indicate active inequality constraints",
    },
    "shakedown_CInEQ": {
        "label": "Shakedown CInEQ (Ilyushin)",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        "symbol": r"C_{\mathrm{ineq}}",
        "formula": r"C_{\mathrm{ineq}}=\max(\varphi_I-1,0)",
        "description": "Ilyushin yield inequality violation",
    },
    "shakedown_layer_residual_stress_SNEG": {
        "label": "Shell layer residual stress (SNEG)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\boldsymbol{\rho}^{S}_{SNEG}",
        "formula": r"\boldsymbol{\rho}^{S}_{k=SNEG}",
        "description": "Layer residual stress on the shell SNEG surface",
    },
    "shakedown_layer_residual_stress_SPOS": {
        "label": "Shell layer residual stress (SPOS)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\rho}^{S}_{SPOS}",
        "formula": r"\boldsymbol{\rho}^{S}_{k=SPOS}",
        "description": "Layer residual stress on the shell SPOS surface",
    },
    "shakedown_layer_total_stress_SNEG": {
        "label": "Shell layer total stress (SNEG)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\boldsymbol{\sigma}^{S}_{tot,SNEG}",
        "formula": r"\boldsymbol{\sigma}^{S}_{v,k}=\boldsymbol{\rho}^{S}_{k}+\alpha\boldsymbol{\sigma}^{E,S}_{v,k}",
        "description": "Layer total stress on the shell SNEG surface at each load-domain vertex",
    },
    "shakedown_layer_total_stress_SPOS": {
        "label": "Shell layer total stress (SPOS)",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\boldsymbol{\sigma}^{S}_{tot,SPOS}",
        "formula": r"\boldsymbol{\sigma}^{S}_{v,k}=\boldsymbol{\rho}^{S}_{k}+\alpha\boldsymbol{\sigma}^{E,S}_{v,k}",
        "description": "Layer total stress on the shell SPOS surface at each load-domain vertex",
    },
    "shakedown_generalized_residual_SF": {
        "label": "Shell generalized SF residual",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\mathbf{SF}_\rho",
        "formula": r"\mathbf{SF}_\rho",
        "description": "Shell residual generalized membrane force",
    },
    "shakedown_generalized_residual_SM": {
        "label": "Shell generalized SM residual",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\mathbf{SM}_\rho",
        "formula": r"\mathbf{SM}_\rho",
        "description": "Shell residual generalized bending moment",
    },
    "shakedown_generalized_total_SF": {
        "label": "Shell generalized SF total",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": True,
        "symbol": r"\mathbf{SF}_{tot}",
        "formula": r"\mathbf{SF}_v=\mathbf{SF}_\rho+\alpha\mathbf{SF}^{E}_v",
        "description": "Shell total generalized membrane force at each load-domain vertex",
    },
    "shakedown_generalized_total_SM": {
        "label": "Shell generalized SM total",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "default_selected": False,
        "symbol": r"\mathbf{SM}_{tot}",
        "formula": r"\mathbf{SM}_v=\mathbf{SM}_\rho+\alpha\mathbf{SM}^{E}_v",
        "description": "Shell total generalized bending moment at each load-domain vertex",
    },
    "shakedown_ilyushin_phi": {
        "label": "Ilyushin yield function",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": True,
        "symbol": r"\varphi_I",
        "formula": r"\varphi_I=\varphi_N+\varphi_M+2c|\varphi_{NM}|",
        "description": "Ilyushin yield function value; values near 1 indicate active inequality constraints",
    },
    "shakedown_ilyushin_active_branch": {
        "label": "Ilyushin active constraint indicator",
        "family": "other",
        "tensor_kind": "scalar",
        "default_selected": False,
        "symbol": r"\mathbb{1}_{\varphi_I\approx1}",
        "formula": r"\mathbb{1}(\varphi_I\ge0.999)",
        "description": "Binary indicator of Ilyushin inequality activity derived from the yield function",
    },
}


def _shakedown_arrays(raw: dict, result_record: Any = None) -> dict[str, Any]:
    """Collect shakedown fields from nested or legacy top-level MAT layouts."""
    return {
        "gauss_coords": _lookup_any(raw, "gauss_coords", result_record),
        "gauss_vols": _lookup_any(raw, "gauss_vols", result_record),
        "gauss_stress_cases": _lookup_any(raw, "gauss_stress_cases", result_record),
        "sigma_E": _lookup_any(raw, "sigma_E", result_record),
        "vertex_load_matrix": _lookup_any(raw, "vertex_load_matrix", result_record),
        "load_factor": _lookup_any(raw, "load_factor", result_record),
        "R_ratios": _lookup_any(raw, "R_ratios", result_record),
        "theta_deg": _lookup_any(raw, "theta_deg", result_record),
        "phi_deg": _lookup_any(raw, "phi_deg", result_record),
        "load_factor_set": _lookup_any(raw, "load_factor_set", result_record),
        "objective_value": _lookup_any(raw, "objective_value", result_record),
        "residual_stress": _lookup_any(raw, "residual_stress", result_record),
        "yield_ratio": _lookup_any(raw, "yield_ratio", result_record),
        "yield_stress": _lookup_any(raw, "yield_stress", result_record),
        "yield_values": _lookup_any(raw, "yield_values", result_record),
        "dual_variables_ineq": _lookup_any(raw, "dual_variables_ineq", result_record),
        "max_effective_excess_per_gp": _lookup_any(raw, "max_effective_excess_per_gp", result_record),
        "family": _lookup_any(raw, "family", result_record),
        "shell_yield": _lookup_any(raw, "shell_yield", result_record),
        "ilyushin_c": _lookup_any(raw, "ilyushin_c", result_record),
        "residual_stress_kind": _lookup_any(raw, "residual_stress_kind", result_record),
        "yield_measure": _lookup_any(raw, "yield_measure", result_record),
        "shell_layer_stress_cases": _lookup_any(raw, "shell_layer_stress_cases", result_record),
        "shell_layer_strain_cases": _lookup_any(raw, "shell_layer_strain_cases", result_record),
        "shell_generalized_stress_cases": _lookup_any(raw, "shell_generalized_stress_cases", result_record),
        "shell_section_z": _lookup_any(raw, "shell_section_z", result_record),
        "shell_elem_thickness": _lookup_any(raw, "shell_elem_thickness", result_record),
        "E": _lookup_any(raw, "E", result_record),
        "nu": _lookup_any(raw, "nu", result_record),
        "residual_generalized_stress": _lookup_any(raw, "residual_generalized_stress", result_record),
        "residual_generalized_strain": _lookup_any(raw, "residual_generalized_strain", result_record),
        "shell_nforc": _lookup_any(raw, "shell_nforc", result_record),
        "shell_force_error": _lookup_any(raw, "shell_force_error", result_record),
        "free_dofs": _lookup_any(raw, "free_dofs", result_record),
        "C_sparse": _lookup_any(raw, "C_sparse", result_record),
        "equilibrium_residual": _lookup_any(raw, "equilibrium_residual", result_record),
        "equilibrium_residual_norm1": _lookup_any(raw, "equilibrium_residual_norm1", result_record),
        "NumVert": _lookup_any(raw, "NumVert", result_record),
    }


def _shakedown_stress_cases(parts: dict[str, Any]) -> Optional[np.ndarray]:
    stress_cases = _as_float_array(parts.get("gauss_stress_cases"), squeeze=False)
    if stress_cases is not None:
        arr = np.asarray(stress_cases, dtype=np.float64)
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape[1] in (3, 6):
            return arr[:, :, np.newaxis]
        if arr.ndim == 3 and arr.shape[1] in (3, 6):
            return arr
        if arr.ndim == 3 and arr.shape[2] in (3, 6):
            return np.moveaxis(arr, 0, 2)

    sigma_e = _as_float_array(parts.get("sigma_E"), squeeze=False)
    if sigma_e is None:
        return None
    arr = np.asarray(sigma_e, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 2 and arr.shape[1] in (3, 6):
        return arr[:, :, np.newaxis]
    if arr.ndim == 3 and arr.shape[1] in (3, 6):
        return arr
    if arr.ndim == 3 and arr.shape[2] in (3, 6):
        return np.moveaxis(arr, 0, 2)
    return None


def _shakedown_text(parts: dict[str, Any], key: str) -> str:
    return _coerce_text(parts.get(key)).strip().lower()


def _shakedown_is_shell_layer(parts: dict[str, Any]) -> bool:
    return (
        _shakedown_text(parts, "shell_yield") == "layer"
        or _shakedown_text(parts, "residual_stress_kind") == "layer"
        or _as_float_array(parts.get("shell_layer_stress_cases"), squeeze=False) is not None
    )


def _shakedown_is_shell_ilyushin(parts: dict[str, Any]) -> bool:
    if _shakedown_is_shell_layer(parts):
        return False
    if _shakedown_text(parts, "shell_yield") == "ilyushin":
        return True
    return _shakedown_text(parts, "residual_stress_kind") == "generalized"


def _shakedown_layer_cases(parts: dict[str, Any]) -> Optional[np.ndarray]:
    value = _as_float_array(parts.get("shell_layer_stress_cases"), squeeze=False)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[:, :, :, np.newaxis]
    if arr.ndim == 4 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return np.moveaxis(arr, -1, 2)
    return None


def _shakedown_layer_surface_index(parts: dict[str, Any], surface: str) -> int:
    cases = _shakedown_layer_cases(parts)
    n_sp = int(cases.shape[1]) if cases is not None and cases.ndim == 4 else 1
    if n_sp <= 1:
        return 0
    z = _as_float_array(parts.get("shell_section_z"))
    if z is not None and z.size >= n_sp:
        flat = z.reshape(-1)[:n_sp]
        return int(np.argmin(flat) if str(surface).upper() == "SNEG" else np.argmax(flat))
    return 0 if str(surface).upper() == "SNEG" else n_sp - 1


def _shakedown_layer_residual_surface(parts: dict[str, Any], surface: str) -> Optional[np.ndarray]:
    cases = _shakedown_layer_cases(parts)
    residual = _as_float_array(parts.get("residual_stress"))
    if cases is None or residual is None:
        return None
    n_surface, n_sp = int(cases.shape[0]), int(cases.shape[1])
    arr = np.asarray(residual, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[:2] == (n_surface, n_sp) and arr.shape[2] == 3:
        layer = arr
    elif arr.ndim == 2 and arr.shape == (n_surface * n_sp, 3):
        layer = arr.reshape(n_surface, n_sp, 3)
    elif arr.ndim == 2 and arr.shape == (n_surface, 3):
        layer = arr[:, np.newaxis, :]
    else:
        return None
    return layer[:, _shakedown_layer_surface_index(parts, surface), :]


def _shakedown_layer_total_surface(parts: dict[str, Any], surface: str) -> Optional[np.ndarray]:
    residual_surface = _shakedown_layer_residual_surface(parts, surface)
    cases = _shakedown_layer_cases(parts)
    vertex_load = _shakedown_vertex_load_matrix(parts)
    alpha = _alpha_scalar(parts.get("objective_value"))
    if residual_surface is None or cases is None or vertex_load is None or alpha is None:
        return None
    k = _shakedown_layer_surface_index(parts, surface)
    elastic_surface_cases = cases[:, k, :, :]
    if elastic_surface_cases.shape[2] != vertex_load.shape[0]:
        return None
    elastic_vertices = np.einsum("gci,iv->gcv", elastic_surface_cases, vertex_load)
    return residual_surface[np.newaxis, :, :] + float(alpha) * np.moveaxis(elastic_vertices, 2, 0)


def _plane_stress_compliance(stress: np.ndarray, *, e_mod: float, nu: float) -> np.ndarray:
    arr = np.asarray(stress, dtype=np.float64)
    if arr.size == 0 or arr.shape[-1] < 3:
        return arr
    e = float(e_mod) if np.isfinite(float(e_mod)) and abs(float(e_mod)) > 0.0 else 1.0
    n = float(nu) if np.isfinite(float(nu)) else 0.3
    out = np.zeros_like(arr, dtype=np.float64)
    out[..., 0] = (arr[..., 0] - n * arr[..., 1]) / e
    out[..., 1] = (arr[..., 1] - n * arr[..., 0]) / e
    out[..., 2] = 2.0 * (1.0 + n) * arr[..., 2] / e
    return out


def _shakedown_layer_yield_surface(parts: dict[str, Any], surface: str) -> Optional[np.ndarray]:
    cases = _shakedown_layer_cases(parts)
    ratio = _as_float_array(parts.get("yield_ratio"))
    if cases is None or ratio is None:
        return None
    n_surface, n_sp = int(cases.shape[0]), int(cases.shape[1])
    arr = np.asarray(ratio, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size == n_surface * n_sp:
            layer = arr.reshape(n_surface, n_sp, 1)
        else:
            return None
    elif arr.ndim == 2:
        if arr.shape[0] == n_surface * n_sp:
            layer = arr.reshape(n_surface, n_sp, arr.shape[1])
        elif arr.shape[0] == n_surface:
            layer = arr[:, np.newaxis, :]
        else:
            return None
    else:
        return None
    return layer[:, _shakedown_layer_surface_index(parts, surface), :].T


def _shell_simpson_weights(thickness: float, n_section_points: int) -> Optional[np.ndarray]:
    if n_section_points != 5:
        return None
    return (float(thickness) / 12.0) * np.asarray([1.0, 4.0, 2.0, 4.0, 1.0], dtype=np.float64)


def _normalize_generalized_cases(value: Any) -> Optional[np.ndarray]:
    arr = _as_float_array(value, squeeze=False)
    if arr is None:
        return None
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim == 2 and arr.shape[1] >= 6:
        return arr[:, :6, np.newaxis]
    if arr.ndim == 3 and arr.shape[1] >= 6:
        return arr[:, :6, :]
    if arr.ndim == 3 and arr.shape[2] >= 6:
        return np.moveaxis(arr[:, :, :6], 2, 1)
    return None


def _integrate_layer_to_generalized(
    layer_values: np.ndarray,
    elem_thickness: np.ndarray,
    section_zeta: np.ndarray,
) -> Optional[np.ndarray]:
    arr = np.asarray(layer_values, dtype=np.float64)
    had_case_axis = arr.ndim == 4
    if arr.ndim == 3:
        arr = arr[:, :, :, np.newaxis]
    if arr.ndim != 4 or arr.shape[2] != 3:
        return None
    n_surface, n_sp, _, n_cases = arr.shape
    zeta = np.asarray(section_zeta, dtype=np.float64).reshape(-1)
    h = np.asarray(elem_thickness, dtype=np.float64).reshape(-1)
    if n_sp != zeta.size or h.size == 0 or n_surface % h.size != 0:
        return None

    n_gp = int(n_surface // h.size)
    out = np.zeros((n_surface, 6, n_cases), dtype=np.float64)
    for elem_idx, h_e in enumerate(h):
        weights = _shell_simpson_weights(float(h_e), n_sp)
        if weights is None:
            return None
        rows = slice(elem_idx * n_gp, (elem_idx + 1) * n_gp)
        z_abs = zeta * float(h_e)
        block = arr[rows]
        out[rows, :3, :] = np.einsum("k,gkcn->gcn", weights, block)
        out[rows, 3:, :] = np.einsum("k,gkcn->gcn", weights * z_abs, block)
    return out if had_case_axis else out[:, :, 0]


def _shakedown_layer_generalized_cases(parts: dict[str, Any]) -> Optional[np.ndarray]:
    direct = _normalize_generalized_cases(parts.get("shell_generalized_stress_cases"))
    if direct is not None:
        return direct
    layer_cases = _shakedown_layer_cases(parts)
    thickness = _as_float_array(parts.get("shell_elem_thickness"))
    section_z = _as_float_array(parts.get("shell_section_z"))
    if layer_cases is None or thickness is None or section_z is None:
        return None
    return _integrate_layer_to_generalized(layer_cases, thickness, section_z)


def _shakedown_layer_residual_generalized(parts: dict[str, Any]) -> Optional[np.ndarray]:
    direct = _normalize_generalized_cases(parts.get("residual_generalized_stress"))
    if direct is not None:
        return direct[:, :, 0]

    layer_cases = _shakedown_layer_cases(parts)
    residual = _as_float_array(parts.get("residual_stress"))
    thickness = _as_float_array(parts.get("shell_elem_thickness"))
    section_z = _as_float_array(parts.get("shell_section_z"))
    if layer_cases is None or residual is None or thickness is None or section_z is None:
        return None

    n_surface, n_sp = int(layer_cases.shape[0]), int(layer_cases.shape[1])
    arr = np.asarray(residual, dtype=np.float64)
    if arr.ndim == 3 and arr.shape == (n_surface, n_sp, 3):
        layer_residual = arr
    elif arr.ndim == 2 and arr.shape == (n_surface * n_sp, 3):
        layer_residual = arr.reshape(n_surface, n_sp, 3)
    else:
        return None
    return _integrate_layer_to_generalized(layer_residual, thickness, section_z)


def _shakedown_generalized_component(parts: dict[str, Any], *, total: bool, suffix: str) -> Optional[np.ndarray]:
    residual = _as_float_array(parts.get("residual_stress"))
    offset = 0 if str(suffix).upper() == "SF" else 3
    if residual is not None and residual.ndim == 2 and residual.shape[1] >= 6:
        residual_gen = residual[:, :6]
    else:
        residual_gen = _shakedown_layer_residual_generalized(parts)
    if residual_gen is None or residual_gen.ndim != 2 or residual_gen.shape[1] < offset + 3:
        return None
    residual_comp = residual_gen[:, offset:offset + 3]
    if not total:
        return residual_comp
    stress_cases = _shakedown_layer_generalized_cases(parts) if _shakedown_is_shell_layer(parts) else _shakedown_stress_cases(parts)
    vertex_load = _shakedown_vertex_load_matrix(parts)
    alpha = _alpha_scalar(parts.get("objective_value"))
    if stress_cases is None or vertex_load is None or alpha is None:
        return None
    if stress_cases.shape[0] != residual_gen.shape[0] or stress_cases.shape[1] < offset + 3:
        return None
    comp_cases = stress_cases[:, offset:offset + 3, :]
    if comp_cases.shape[2] != vertex_load.shape[0]:
        return None
    elastic_vertices = np.einsum("gci,iv->gcv", comp_cases, vertex_load)
    return residual_comp[np.newaxis, :, :] + float(alpha) * np.moveaxis(elastic_vertices, 2, 0)


def _shakedown_yield_ratio_frames(parts: dict[str, Any], n_surface: int | None = None) -> Optional[np.ndarray]:
    ratio = _as_float_array(parts.get("yield_ratio"))
    if ratio is None:
        return None
    arr = np.asarray(ratio, dtype=np.float64)
    if arr.ndim == 1:
        if n_surface is not None and n_surface > 0 and arr.size % n_surface == 0:
            return arr.reshape(n_surface, -1).max(axis=1).reshape(1, n_surface)
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    if n_surface is not None and n_surface > 0 and arr.shape[0] != n_surface and arr.shape[0] % n_surface == 0:
        n_sp = int(arr.shape[0] // n_surface)
        return arr.reshape(n_surface, n_sp, arr.shape[1]).max(axis=1).T
    return arr.T


def _shakedown_counts(parts: dict[str, Any]) -> tuple[int, int, int]:
    residual = _as_float_array(parts.get("residual_stress"))
    stress_cases = _shakedown_stress_cases(parts)
    yield_ratio = _as_float_array(parts.get("yield_ratio"))
    vertex_load = _shakedown_vertex_load_matrix(parts)
    layer_cases = _shakedown_layer_cases(parts)

    n_gauss = 0
    n_str = 1
    n_vert = 1
    if layer_cases is not None:
        n_gauss = int(layer_cases.shape[0])
        n_str = 3
    elif residual is not None and residual.ndim >= 2:
        n_gauss = int(residual.shape[0])
        n_str = int(residual.shape[1])
    elif stress_cases is not None and stress_cases.ndim >= 3:
        n_gauss = int(stress_cases.shape[0])
        n_str = int(stress_cases.shape[1])
    elif yield_ratio is not None and yield_ratio.ndim >= 2:
        n_gauss = int(yield_ratio.shape[0])

    if yield_ratio is not None and yield_ratio.ndim >= 2:
        n_vert = int(yield_ratio.shape[1])
    elif vertex_load is not None and vertex_load.ndim >= 2:
        n_vert = int(vertex_load.shape[1])
    else:
        num_vert = _as_float_array(parts.get("NumVert"))
        if num_vert is not None and num_vert.size:
            n_vert = int(num_vert.reshape(-1)[0])

    return n_gauss, n_str, max(1, n_vert)


def _free_dofs_array(parts: dict[str, Any]) -> Optional[np.ndarray]:
    value = parts.get("free_dofs")
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


def _shakedown_equilibrium_node_shape(parts: dict[str, Any]) -> tuple[int, int]:
    free_dofs = _free_dofs_array(parts)
    if free_dofs is None:
        return 0, 0
    max_dof = int(np.max(free_dofs))
    n_comp = 3
    coords = _as_float_array(parts.get("gauss_coords"))
    if coords is not None and coords.ndim >= 1 and coords.shape[-1] in (2, 3):
        n_comp = int(coords.shape[-1])
    n_nodes = max_dof // n_comp + 1
    return int(n_nodes), int(n_comp)


def _shakedown_vertex_load_matrix(parts: dict[str, Any]) -> Optional[np.ndarray]:
    direct = _as_float_array(parts.get("vertex_load_matrix"))
    if direct is not None and direct.ndim == 2 and direct.size:
        return direct

    stress_cases = _shakedown_stress_cases(parts)
    layer_cases = _shakedown_layer_cases(parts)
    if stress_cases is not None and stress_cases.ndim == 3:
        n_independent = int(stress_cases.shape[2])
    elif layer_cases is not None and layer_cases.ndim == 4:
        n_independent = int(layer_cases.shape[3])
    else:
        return None
    if n_independent < 1 or n_independent > 3:
        return None

    num_vert = _as_float_array(parts.get("NumVert"))
    n_vertices = int(num_vert.reshape(-1)[0]) if num_vert is not None and num_vert.size else None

    load_factor_set = _as_float_array(parts.get("load_factor_set"))
    if load_factor_set is not None and load_factor_set.ndim == 2 and load_factor_set.size:
        rows = load_factor_set[:, :n_independent]
        if n_vertices is None or rows.shape[0] == n_vertices:
            return np.asarray(rows.T, dtype=np.float64)

    load_scale = None
    load_factor = _as_float_array(parts.get("load_factor"))
    if load_factor is not None and load_factor.size >= n_independent:
        load_scale = load_factor.reshape(-1)[:n_independent]

    theta = _alpha_scalar(parts.get("theta_deg"))
    phi = _alpha_scalar(parts.get("phi_deg")) if n_independent == 3 else None
    r_ratios = _as_float_array(parts.get("R_ratios"))
    if r_ratios is None or not r_ratios.size:
        r_ratios = np.zeros(n_independent, dtype=np.float64)
    try:
        from jaxmech.optimize.cvxpy.load_domain import build_vertex_load_matrix

        _, vertex_load, _ = build_vertex_load_matrix(
            n_independent=n_independent,
            load_scale=load_scale,
            n_vertices=n_vertices,
            theta_deg=theta,
            phi_deg=phi,
            R_ratios=r_ratios.reshape(-1)[:n_independent],
        )
        return np.asarray(vertex_load, dtype=np.float64)
    except Exception:
        return _build_vertex_load_matrix_fallback(
            n_independent=n_independent,
            load_scale=load_scale,
            n_vertices=n_vertices,
            theta_deg=theta,
            phi_deg=phi,
            R_ratios=r_ratios.reshape(-1)[:n_independent],
        )


def _build_vertex_load_matrix_fallback(
    *,
    n_independent: int,
    load_scale: Optional[np.ndarray],
    n_vertices: Optional[int],
    theta_deg: Optional[float],
    phi_deg: Optional[float],
    R_ratios: np.ndarray,
) -> Optional[np.ndarray]:
    """Small local fallback for legacy shakedown visualization.

    Importing ``jaxmech.optimize.cvxpy`` can pull optional shell dependencies on
    Windows Web paths.  Visualization only needs the deterministic load-domain
    vertex matrix, so keep this dependency-free copy for MAT recovery.
    """
    if n_independent < 1 or n_independent > 3:
        return None
    if n_vertices is None:
        n_vertices = {1: 2, 2: 4, 3: 8}[n_independent]
    n_vertices = int(n_vertices)

    if n_independent == 1:
        load_point = np.asarray([1.0], dtype=np.float64)
    elif load_scale is not None and np.asarray(load_scale).size >= n_independent:
        load_point = np.asarray(load_scale, dtype=np.float64).reshape(-1)[:n_independent]
    elif n_independent == 2:
        theta = 45.0 if theta_deg is None else float(theta_deg)
        theta_rad = np.deg2rad(theta)
        load_point = np.asarray([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64)
    else:
        theta = 45.0 if theta_deg is None else float(theta_deg)
        phi = float(np.degrees(np.arcsin(1.0 / np.sqrt(3.0)))) if phi_deg is None else float(phi_deg)
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        load_point = np.asarray(
            [
                np.cos(phi_rad) * np.cos(theta_rad),
                np.cos(phi_rad) * np.sin(theta_rad),
                np.sin(phi_rad),
            ],
            dtype=np.float64,
        )

    r_vec = np.asarray(R_ratios, dtype=np.float64).reshape(-1)
    r_full = np.zeros(n_independent, dtype=np.float64)
    if r_vec.size:
        take = min(r_vec.size, n_independent)
        r_full[:take] = r_vec[:take]
    r_max = load_point
    r_min = r_full * load_point
    if n_vertices == 1:
        return r_max.reshape(n_independent, 1)
    if n_vertices == 2:
        return np.column_stack([r_min, r_max])
    if n_vertices == 2 ** n_independent:
        corners = np.asarray(list(np.ndindex(*(2,) * n_independent)), dtype=np.float64).T
        return r_min[:, None] + (r_max - r_min)[:, None] * corners
    return None


def _shakedown_inequality_multiplier_frames(parts: dict[str, Any], n_gauss: int, n_vert: int) -> Optional[np.ndarray]:
    dual = _as_float_array(parts.get("dual_variables_ineq"))
    if dual is None or n_gauss <= 0:
        return None
    arr = np.asarray(dual, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        flat = arr.reshape(-1)
        expected = int(n_gauss) * int(max(1, n_vert))
        layer_cases = _shakedown_layer_cases(parts)
        if layer_cases is not None and layer_cases.ndim == 4:
            n_sp = int(layer_cases.shape[1])
            layer_expected = int(n_gauss) * n_sp * int(max(1, n_vert))
            if flat.size == layer_expected:
                return flat.reshape(int(n_gauss), n_sp, int(max(1, n_vert))).max(axis=1).T
        if flat.size == expected:
            return flat.reshape(int(max(1, n_vert)), int(n_gauss))
        if flat.size == int(n_gauss):
            return flat.reshape(1, int(n_gauss))
        if flat.size > expected and expected > 0:
            return flat[:expected].reshape(int(max(1, n_vert)), int(n_gauss))
        return None
    if arr.ndim == 2:
        if arr.shape == (int(max(1, n_vert)), int(n_gauss)):
            return arr
        if arr.shape == (int(n_gauss), int(max(1, n_vert))):
            return arr.T
        if arr.shape[0] == int(n_gauss):
            return arr.T
        if arr.shape[1] == int(n_gauss):
            return arr
    return None


def _inspect_shakedown_fields(mat_path: Path) -> list[dict]:
    """Inspect nested shakedown result structs and expose derived field entries."""
    import scipy.io as sio

    try:
        raw = sio.loadmat(
            str(mat_path),
            squeeze_me=False,
            struct_as_record=True,
            variable_names=[
                "ElasticInputSet",
                "ResultsSet",
                "ConfigInfo",
                "gauss_coords",
                "gauss_vols",
                "gauss_stress_cases",
                "sigma_E",
                "vertex_load_matrix",
                "load_factor",
                "R_ratios",
                "theta_deg",
                "phi_deg",
                "load_factor_set",
                "objective_value",
                "residual_stress",
                "yield_ratio",
                "dual_variables_ineq",
                "max_effective_excess_per_gp",
                "family",
                "shell_yield",
                "ilyushin_c",
                "residual_stress_kind",
                "yield_measure",
                "shell_layer_stress_cases",
                "shell_layer_strain_cases",
                "shell_generalized_stress_cases",
                "shell_section_z",
                "shell_elem_thickness",
                "residual_generalized_stress",
                "free_dofs",
                "C_sparse",
                "equilibrium_residual",
                "equilibrium_residual_norm1",
                "NumVert",
            ],
        )
    except Exception:
        return []

    parts = _shakedown_arrays(raw)
    n_results = max(1, len(_resultset_records(raw.get("ResultsSet"))))
    n_gauss, n_str, n_vert = _shakedown_counts(parts)
    if n_gauss <= 0:
        return []

    is_layer = _shakedown_is_shell_layer(parts)
    is_ilyushin = _shakedown_is_shell_ilyushin(parts)

    available: set[str] = set()
    if _as_float_array(parts.get("residual_stress")) is not None and not is_layer and not is_ilyushin:
        available.add("shakedown_residual_stress")
    if (
        "shakedown_residual_stress" in available
        and _shakedown_stress_cases(parts) is not None
        and _shakedown_vertex_load_matrix(parts) is not None
        and _as_float_array(parts.get("objective_value")) is not None
    ):
        available.add("shakedown_total_stress")
    if _as_float_array(parts.get("yield_ratio")) is not None:
        if is_ilyushin:
            available.add("shakedown_Phi")
            available.add("shakedown_CInEQ")
        elif not is_layer:
            available.add("shakedown_inequality_violation")
    if is_layer:
        for surface in ("SNEG", "SPOS"):
            if _shakedown_layer_residual_surface(parts, surface) is not None:
                available.add(f"shakedown_S_res_{surface}")
            if _shakedown_layer_total_surface(parts, surface) is not None:
                available.add(f"shakedown_S_tot_{surface}")
            if _shakedown_layer_yield_surface(parts, surface) is not None:
                available.add(f"shakedown_CInEQ_{surface}")
    if is_ilyushin or is_layer:
        for suffix in ("SF", "SM"):
            if _shakedown_generalized_component(parts, total=False, suffix=suffix) is not None:
                available.add(f"shakedown_{suffix}_res")
            if _shakedown_generalized_component(parts, total=True, suffix=suffix) is not None:
                available.add(f"shakedown_{suffix}_tot")
    if _shakedown_inequality_multiplier_frames(parts, n_gauss, n_vert) is not None:
        available.add("shakedown_inequality_multiplier")
    if _as_float_array(parts.get("max_effective_excess_per_gp")) is not None:
        available.add("max_effective_excess_per_gp")
    eq_n_nodes, eq_n_comp = _shakedown_equilibrium_node_shape(parts)
    can_make_eq = (
        eq_n_nodes > 0
        and _free_dofs_array(parts) is not None
        and _compute_shakedown_equilibrium_residual(parts) is not None
    )
    if can_make_eq:
        available.add("shakedown_equality_violation")

    fields: list[dict] = []
    for key in available:
        spec = _SHAKEDOWN_FIELD_SPECS[key]
        if key == "shakedown_equality_violation":
            n_components = eq_n_comp
            location = "node"
            shape = [int(eq_n_nodes), int(n_components)]
        elif key == "shakedown_residual_stress":
            n_components = n_str
            location = "gauss"
            shape = [int(n_gauss), int(n_components)]
        elif key == "max_effective_excess_per_gp":
            n_components = 1
            location = "gauss"
            shape = [int(n_results), int(n_gauss)] if n_results > 1 else [int(n_gauss)]
        elif key == "shakedown_inequality_multiplier":
            n_components = 1
            location = "gauss"
            shape = [int(n_vert), int(n_gauss)]
        elif key.startswith("shakedown_S_res_") or key in {"shakedown_SF_res", "shakedown_SM_res"}:
            n_components = 3
            location = "gauss"
            shape = [int(n_results), int(n_gauss), int(n_components)]
        elif key.startswith("shakedown_layer_residual_stress_") or key.startswith("shakedown_generalized_residual_"):
            n_components = 3
            location = "gauss"
            shape = [int(n_gauss), int(n_components)]
        elif key.startswith("shakedown_S_tot_") or key.startswith("shakedown_layer_total_stress_") or key in {"shakedown_SF_tot", "shakedown_SM_tot"} or key.startswith("shakedown_generalized_total_"):
            n_components = 3
            location = "gauss"
            shape = [int(n_vert), int(n_gauss), int(n_components)]
        elif key.startswith("shakedown_CInEQ_") or key in {"shakedown_Phi", "shakedown_CInEQ"} or key.startswith("shakedown_ilyushin_"):
            n_components = 1
            location = "gauss"
            shape = [int(n_vert), int(n_gauss)]
        else:
            n_components = n_str if "stress" in key else 1
            location = "gauss"
            shape = [int(n_vert), int(n_gauss), int(n_components)] if n_components > 1 else [int(n_vert), int(n_gauss)]
        frames = (
            "single"
            if key in {"shakedown_residual_stress", "shakedown_equality_violation"}
            or key.startswith("shakedown_layer_residual_stress_")
            or key.startswith("shakedown_generalized_residual_")
            else "frames"
        )
        fields.append({
            "key": key,
            "label": spec["label"],
            "shape": shape,
            "mat_class": "derived",
            "location": location,
            "frames": frames,
            "n_components": int(n_components),
            "family": spec["family"],
            "tensor_kind": spec["tensor_kind"] if n_components in (3, 6) else ("scalar" if n_components == 1 else "components"),
            "default_selected": bool(spec["default_selected"]),
            "visualizable": True,
            "symbol": str(spec.get("symbol", "")),
            "formula": str(spec.get("formula", "")),
            "description": str(spec.get("description", "")),
        })
    return fields


def inspect_mat_fields(mat_path: str | Path) -> dict:
    """Inspect a MAT file and return a lightweight visual field catalog.

    This uses ``scipy.io.whosmat`` so large result arrays are not loaded during
    the selection step.  The output is intentionally conservative for unknown
    arrays: known jaxmech field names get precise metadata, while arbitrary
    numeric arrays are listed as ``other`` with location inferred from naming
    and shape where possible.
    """
    import scipy.io as sio

    mat_path = Path(mat_path)
    entries = sio.whosmat(str(mat_path))
    nested_shapes = _load_nested_schema_shapes(mat_path)
    entries_for_schema = _entries_with_nested_schema_shapes(entries, nested_shapes)

    manifest = load_viz_manifest_from_mat(mat_path)
    if manifest is None:
        manifest = _build_legacy_manifest(mat_path, entries_for_schema)
        if manifest is not None:
            entries = sio.whosmat(str(mat_path))
            nested_shapes = _load_nested_schema_shapes(mat_path)
            entries_for_schema = _entries_with_nested_schema_shapes(entries, nested_shapes)
    manifest = _rebuild_rsdm_shakedown_manifest_if_needed(mat_path, manifest, entries_for_schema)
    manifest = _normalize_manifest_for_reader(manifest, [name for name, _shape, _mat_class in entries_for_schema])
    if _whos_has_validation(entries_for_schema):
        return inspect_validation_fields(mat_path, entries_for_schema)
    if manifest is not None:
        if str(manifest.get("analysis_type", "")).strip().lower() == "shakedown":
            supplemental = _inspect_shakedown_fields(mat_path)
            fields = [
                field for field in _fields_from_manifest(manifest, entries_for_schema)
                if str(field.get("key", "")) not in _SHAKEDOWN_FIELD_SPECS
            ]
            seen = {str(field.get("key", "")) for field in fields}
            for field in supplemental:
                if str(field.get("key", "")) not in seen:
                    fields.append(field)
                    seen.add(str(field.get("key", "")))
            fields.sort(key=_field_sort_key)
        else:
            fields = _fields_from_manifest(manifest, entries_for_schema)
        return {
            "source": str(mat_path),
            "manifest": manifest,
            "fields": fields,
        }

    field_context = _infer_field_context_from_whos(entries_for_schema)
    field_context = _augment_field_context_from_inpdata(mat_path, field_context, entries_for_schema)

    bookkeeping_keys = {
        "InpData",
        VIZ_MANIFEST_KEY,
        "metadata",
        "n_gauss_per_elem",
        "frame_time",
        "time_grid",
        "gauss_coords",
        "gauss_vols",
        "free_dofs",
        "fixed_dofs",
        "solid_elem_types",
        "solid_node_labels",
        "solid_element_labels",
    }

    fields: list[dict] = []
    for name, shape, mat_class in entries_for_schema:
        if name.startswith("__") or name.startswith("viz_"):
            continue
        if name in _HIDDEN_VISUAL_FIELDS:
            continue
        if name in bookkeeping_keys:
            continue
        if not _is_numeric_mat_class(mat_class) or not _is_visualizable_shape(shape):
            continue

        descriptor = _KNOWN_FIELD_DESCRIPTORS.get(name)
        layout = _match_field_layout(name, shape, descriptor, field_context)
        if layout is None:
            continue

        if descriptor:
            label, location, is_frames = descriptor
        else:
            label = name
            location = str(layout["location"])
            is_frames = bool(layout["frames"])

        family = _field_family(name, label)
        n_components = int(layout["n_components"])
        tensor_kind = _tensor_kind(family, n_components, name)
        default_selected = (
            family in {"stress", "displacement"}
            or (family == "strain" and "peeq" not in name.lower())
            or name.lower() in {
                "internal_force",
                "frame_internal_force",
                "frame_nforc",
                "frame_force_error",
                "solid_nforc_nodal",
            }
        )

        fields.append({
            "key": name,
            "label": label,
            "shape": [int(v) for v in shape],
            "mat_class": mat_class,
            "location": location,
            "frames": "frames" if is_frames else "single",
            "n_components": int(n_components),
            "family": family,
            "tensor_kind": tensor_kind,
            "default_selected": bool(default_selected),
            "visualizable": True,
        })

    fields.extend(_inspect_shakedown_fields(mat_path))
    fields.sort(key=_field_sort_key)
    return {
        "source": str(mat_path),
        "fields": fields,
    }


def _register_single_field(
    raw: dict,
    vd: VizData,
    keys: tuple[str, ...],
    label: str,
    loc: str,
    selected_fields: Optional[set[str]] = None,
) -> None:
    for key in keys:
        if selected_fields is not None and key not in selected_fields:
            continue
        arr = raw.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        if loc == "node":
            arr = _reshape_nodal_field_if_flat(arr, vd)
        n_comp = _infer_component_count(arr, vd, loc)
        display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
        vd.fields[key] = arr
        vd.field_info[key] = FieldInfo(
            key=key, label=label, n_components=n_comp,
            location=loc, n_frames=1,
            symbol=display_meta.get("symbol", ""),
            formula=display_meta.get("formula", ""),
            description=display_meta.get("description", ""),
        )
        return


def _register_multiframe_field(
    raw: dict,
    vd: VizData,
    keys: tuple[str, ...],
    label: str,
    loc: str,
    selected_fields: Optional[set[str]] = None,
) -> None:
    for key in keys:
        if selected_fields is not None and key not in selected_fields:
            continue
        arr = raw.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim < 2 or arr.size == 0:
            continue
        if loc == "node":
            arr = _reshape_nodal_field_if_flat(arr, vd)
        n_frames = int(arr.shape[0])
        n_comp = _infer_component_count(arr[0], vd, loc)
        display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
        vd.fields[key] = arr
        vd.field_info[key] = FieldInfo(
            key=key, label=label, n_components=n_comp,
            location=loc, n_frames=n_frames,
            symbol=display_meta.get("symbol", ""),
            formula=display_meta.get("formula", ""),
            description=display_meta.get("description", ""),
        )
        return


def _discover_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]] = None) -> None:
    """Populate ``vd.fields`` and ``vd.field_info`` from *raw*."""

    for keys, label, loc in _SINGLE_FRAME_FIELDS:
        _register_single_field(raw, vd, keys, label, loc, selected_fields)

    for keys, label, loc in _MULTI_FRAME_FIELDS:
        _register_multiframe_field(raw, vd, keys, label, loc, selected_fields)


def _infer_extra_field_info(key: str, arr: np.ndarray, vd: VizData) -> Optional[FieldInfo]:
    """Infer FieldInfo for a selected field not covered by known patterns."""
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None

    family = _field_family(key)
    lowered = key.lower()
    is_frames = lowered.startswith("frame_") or lowered.endswith("_cycle")
    location = "gauss" if family in {"stress", "strain"} else "node" if family in {"displacement", "force"} else "gauss"
    if "nodal" in lowered or "nforc" in lowered or "reaction" in lowered:
        location = "node"
    elif "gauss" in lowered:
        location = "gauss"

    n_frames = int(arr.shape[0]) if is_frames and arr.ndim >= 2 else 1
    sample = arr[0] if n_frames > 1 else arr
    n_comp = _infer_component_count(sample, vd, location)
    display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
    return FieldInfo(
        key=key,
        label=key,
        n_components=n_comp,
        location=location,
        n_frames=n_frames,
        symbol=display_meta.get("symbol", ""),
        formula=display_meta.get("formula", ""),
        description=display_meta.get("description", ""),
    )


def _register_selected_extra_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    if not selected_fields:
        return
    for key in sorted(selected_fields):
        if key in vd.fields or key not in raw:
            continue
        if key in _HIDDEN_VISUAL_FIELDS:
            continue
        arr = np.asarray(raw[key])
        fi = _infer_extra_field_info(key, arr, vd)
        if fi is None:
            continue
        if fi.location == "node":
            arr = _reshape_nodal_field_if_flat(arr, vd)
            fi = FieldInfo(
                key=fi.key,
                label=fi.label,
                n_components=_infer_component_count(arr[0] if fi.n_frames > 1 and arr.ndim >= 2 else arr, vd, "node"),
                location=fi.location,
                n_frames=int(arr.shape[0]) if fi.n_frames > 1 and arr.ndim >= 2 else fi.n_frames,
                symbol=fi.symbol,
                formula=fi.formula,
                description=fi.description,
            )
        vd.fields[key] = arr.astype(np.float64, copy=False)
        vd.field_info[key] = fi


def _extract_frame_times(raw: dict) -> Optional[np.ndarray]:
    for key in ("validation_frame_time", "abaqus_validation_frame_time", "frame_time", "time_grid", "frame_load_scale"):
        arr = raw.get(key)
        if arr is not None:
            return _squeeze(np.asarray(arr, dtype=np.float64))
    return None


_SHELL_DERIVED_SURFACE_STRAIN_KEYS = {
    "frame_shell_layer_strain_SNEG",
    "frame_shell_layer_strain_SPOS",
}


def _selected_field_dependencies(selected_fields: set[str]) -> set[str]:
    deps: set[str] = set()
    if selected_fields & _SHELL_DERIVED_SURFACE_STRAIN_KEYS:
        deps.update({
            "frame_gauss_strain",
            "shell_section_z",
            "shell_surface_section_indices",
            "shell_surface_names",
        })
    if "rsdm_XS_path" in selected_fields:
        deps.update({"rsdm_XS_cyc", "excess_stress_cycle", "field_excess_stress_cycle", "time_grid", "field_time_grid", "RSDMShakedownInput"})
    if selected_fields & {"rsdms_XS_path", "rsdms_XS_path_SNEG", "rsdms_XS_path_SPOS"}:
        deps.update({"ResultsSet", "ConfigInfo", "yield_ratio", "vertex_load_matrix", "shell_layer_stress_cases", "shell_section_z"})
    return deps


def _derive_shell_surface_strain(raw: Mapping[str, Any], surface: str) -> np.ndarray | None:
    gen_raw = raw.get("frame_gauss_strain")
    z_raw = raw.get("shell_section_z")
    if gen_raw is None or z_raw is None:
        return None
    gen = np.asarray(gen_raw, dtype=np.float64)
    if gen.ndim < 4 or gen.shape[-1] < 6:
        return None
    z_values = np.asarray(z_raw, dtype=np.float64).reshape(-1)
    if z_values.size == 0:
        return None
    indices_raw = raw.get("shell_surface_section_indices")
    if indices_raw is not None:
        indices = np.asarray(indices_raw, dtype=np.int32).reshape(-1)
    else:
        indices = np.asarray([0, max(z_values.size - 1, 0)], dtype=np.int32)
    if indices.size < 2:
        indices = np.asarray([0, max(z_values.size - 1, 0)], dtype=np.int32)
    surface_key = str(surface).upper()
    idx = int(indices[0] if surface_key == "SNEG" else indices[1])
    idx = max(0, min(idx, z_values.size - 1))
    z = float(z_values[idx])
    return np.asarray(gen[..., :3] + z * gen[..., 3:6], dtype=np.float64)


def _register_shell_incremental_derived_fields(
    raw: Mapping[str, Any],
    vd: VizData,
    selected_fields: Optional[set[str]],
) -> None:
    for surface in ("SNEG", "SPOS"):
        key = f"frame_shell_layer_strain_{surface}"
        if selected_fields is not None and key not in selected_fields:
            continue
        if key in vd.fields:
            continue
        arr = _derive_shell_surface_strain(raw, surface)
        if arr is None or arr.size == 0:
            continue
        spec = _incremental_shell_display_spec(key)
        vd.fields[key] = arr
        vd.field_info[key] = FieldInfo(
            key=key,
            label=spec.get("label", f"E shell layer strain {surface} history"),
            n_components=3,
            location="gauss",
            n_frames=int(arr.shape[0]) if arr.ndim >= 2 else 1,
            symbol=spec.get("symbol", ""),
            formula=spec.get("formula", ""),
            description=spec.get("description", ""),
        )


_VALIDATION_SIDE_PREFIX = {
    "jax": "validation_jax_",
    "abaqus": "validation_abaqus_",
}

_VALIDATION_DISPLAY: dict[str, dict[str, Any]] = {
    "U": {
        "label": "U displacement",
        "display_key": "U",
        "family": "displacement",
        "tensor_kind": "vector",
        "location": "node",
        "symbol": r"\mathbf{u}(t)",
        "formula": r"\mathbf{u}^{JAX}(t)\;\leftrightarrow\;\mathbf{u}^{ODB}(t)",
        "description": "Nodal displacement history used for JAX/ABAQUS ODB validation.",
    },
    "NFORC": {
        "label": "NFORC nodal force",
        "display_key": "NFORC",
        "family": "force",
        "tensor_kind": "vector",
        "location": "node",
        "symbol": r"\mathbf{NFORC}(t)",
        "formula": r"\mathbf{NFORC}(t)=\int_{\Omega_e}\mathbf{B}^{T}\boldsymbol{\sigma}(t)\,d\Omega",
        "description": "Nodal force due to stress, compared against ABAQUS NFORC output.",
    },
    "S": {
        "label": "S stress",
        "display_key": "S",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\sigma}(t)",
        "formula": r"\boldsymbol{\sigma}^{JAX}(t)\;\leftrightarrow\;\boldsymbol{\sigma}^{ODB}(t)",
        "description": "Gauss-point stress history used for ODB validation.",
    },
    "E": {
        "label": "E strain",
        "display_key": "E",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{JAX}(t)\;\leftrightarrow\;\boldsymbol{\varepsilon}^{ODB}(t)",
        "description": "Gauss-point strain history used for ODB validation.",
    },
    "S_layer_SNEG": {
        "label": "S shell layer stress SNEG history",
        "display_key": "S_SNEG",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\sigma}^{S}_{SNEG}(t)",
        "formula": r"\boldsymbol{\sigma}^{S}_{SNEG}(t)=\boldsymbol{\sigma}^{S}(z_{SNEG},t)=[S_{11},S_{22},S_{12}]",
        "description": "History of plane-stress Cauchy layer stress on the shell SNEG surface.",
    },
    "S_layer_SPOS": {
        "label": "S shell layer stress SPOS history",
        "display_key": "S_SPOS",
        "family": "stress",
        "tensor_kind": "stress_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\sigma}^{S}_{SPOS}(t)",
        "formula": r"\boldsymbol{\sigma}^{S}_{SPOS}(t)=\boldsymbol{\sigma}^{S}(z_{SPOS},t)=[S_{11},S_{22},S_{12}]",
        "description": "History of plane-stress Cauchy layer stress on the shell SPOS surface.",
    },
    "E_layer_SNEG": {
        "label": "E shell layer strain SNEG history",
        "display_key": "E_SNEG",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}^{S}_{SNEG}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{S}_{SNEG}(t)=\boldsymbol{\varepsilon}^{S}(z_{SNEG},t)=[E_{11},E_{22},E_{12}]",
        "description": "History of shell layer strain on the SNEG surface.",
    },
    "E_layer_SPOS": {
        "label": "E shell layer strain SPOS history",
        "display_key": "E_SPOS",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}^{S}_{SPOS}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{S}_{SPOS}(t)=\boldsymbol{\varepsilon}^{S}(z_{SPOS},t)=[E_{11},E_{22},E_{12}]",
        "description": "History of shell layer strain on the SPOS surface.",
    },
    "PE_layer_SNEG": {
        "label": "PE shell layer plastic strain SNEG history",
        "display_key": "PE_SNEG",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}^{p,S}_{SNEG}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{p,S}_{SNEG}(t)=\boldsymbol{\varepsilon}^{p,S}(z_{SNEG},t)=[PE_{11},PE_{22},PE_{12},PE_{33}]",
        "description": "History of shell layer plastic strain on the SNEG surface.",
    },
    "PE_layer_SPOS": {
        "label": "PE shell layer plastic strain SPOS history",
        "display_key": "PE_SPOS",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}^{p,S}_{SPOS}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{p,S}_{SPOS}(t)=\boldsymbol{\varepsilon}^{p,S}(z_{SPOS},t)=[PE_{11},PE_{22},PE_{12},PE_{33}]",
        "description": "History of shell layer plastic strain on the SPOS surface.",
    },
    "PEEQ": {
        "label": "PEEQ",
        "display_key": "PEEQ",
        "family": "strain",
        "tensor_kind": "scalar",
        "location": "gauss",
        "symbol": r"\bar{\varepsilon}^{p}(t)",
        "formula": r"\bar{\varepsilon}^{p}(t)=\int_0^t\sqrt{\frac{2}{3}\dot{\boldsymbol{\varepsilon}}^p:\dot{\boldsymbol{\varepsilon}}^p}\,d\tau",
        "description": "Equivalent plastic strain used for elastoplastic or DCA validation.",
    },
    "PE": {
        "label": "PE plastic strain",
        "display_key": "PE",
        "family": "strain",
        "tensor_kind": "strain_voigt",
        "location": "gauss",
        "symbol": r"\boldsymbol{\varepsilon}^{p}(t)",
        "formula": r"\boldsymbol{\varepsilon}^{p,JAX}(t)\;\leftrightarrow\;\boldsymbol{\varepsilon}^{p,ODB}(t)",
        "description": "Gauss-point plastic strain history using the ABAQUS PE convention for solid elastoplastic or DCA validation.",
    },
    "PEEQ_Neg": {
        "label": "PEEQ shell layer SNEG history",
        "display_key": "PEEQ_Neg",
        "family": "strain",
        "tensor_kind": "scalar",
        "location": "gauss",
        "symbol": r"\bar{\varepsilon}^{p}_{SNEG}(t)",
        "formula": r"\bar{\varepsilon}^{p}_{SNEG}(t)=\bar{\varepsilon}^{p}(z_{SNEG},t)",
        "description": "History of equivalent plastic strain on the shell SNEG surface.",
    },
    "PEEQ_Pos": {
        "label": "PEEQ shell layer SPOS history",
        "display_key": "PEEQ_Pos",
        "family": "strain",
        "tensor_kind": "scalar",
        "location": "gauss",
        "symbol": r"\bar{\varepsilon}^{p}_{SPOS}(t)",
        "formula": r"\bar{\varepsilon}^{p}_{SPOS}(t)=\bar{\varepsilon}^{p}(z_{SPOS},t)",
        "description": "History of equivalent plastic strain on the shell SPOS surface.",
    },
    "PEEQ_Max": {
        "label": "PEEQ through-thickness max history",
        "display_key": "PEEQ_Max",
        "family": "strain",
        "tensor_kind": "scalar",
        "location": "gauss",
        "symbol": r"\bar{\varepsilon}^{p}_{\max}(t)",
        "formula": r"\bar{\varepsilon}^{p}_{\max}(t)=\max_{z_k}\bar{\varepsilon}^{p}(z_k,t)",
        "description": "History of the maximum equivalent plastic strain over the exposed shell section surfaces.",
    },
}

_VALIDATION_METADATA_KEYS = (
    "validation_abaqus_surface_section_indices",
    "validation_abaqus_source_surface_section_indices",
    "validation_abaqus_surface_section_numbers",
    "validation_abaqus_all_section_numbers",
    "validation_abaqus_surface_section_names",
    "validation_abaqus_section_count",
    "validation_abaqus_all_section_count",
    "validation_jax_section_count",
)


def _validation_is_shell(raw: Mapping[str, Any]) -> bool:
    if any(key in raw for key in ("shell_u_nodal", "shell_gen_internal_force", "shell_layer_stress", "shell_layer_strain")):
        return True
    if any(key in raw for key in ("abaqus_gen_section_force", "abaqus_gen_section_moment", "abaqus_layer_stress", "abaqus_layer_strain")):
        return True
    if any(
        key in raw
        for key in (
            "validation_jax_shell_layer_s",
            "validation_abaqus_shell_layer_s",
            "validation_jax_shell_layer_e",
            "validation_abaqus_shell_layer_e",
            "frame_shell_layer_stress_SNEG",
            "frame_shell_layer_stress_SPOS",
            "frame_shell_layer_plastic_strain_SNEG",
            "frame_shell_layer_plastic_strain_SPOS",
            "frame_shell_layer_peeq_SNEG",
            "frame_shell_layer_peeq_SPOS",
            "abaqus_frame_shell_layer_stress_SNEG",
            "abaqus_frame_shell_layer_stress_SPOS",
        )
    ):
        return True
    meta = _extract_metadata(dict(raw)) if isinstance(raw, dict) else {}
    return str(meta.get("family", "")).strip().lower() == "shell"


def _validation_flat_ints(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        arr = np.asarray(value).reshape(-1)
    except Exception:
        return []
    out: list[int] = []
    for item in arr:
        try:
            out.append(int(np.asarray(item).reshape(-1)[0]))
        except Exception:
            continue
    return out


def _validation_flat_strings(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=object).reshape(-1)
    except Exception:
        return []
    out: list[str] = []
    for item in arr:
        cur = _unwrap_scalar(item)
        if isinstance(cur, bytes):
            text = cur.decode("utf-8", errors="ignore").strip()
        else:
            text = str(cur).strip()
        if text:
            out.append(text)
    return out


def _load_validation_display_metadata(mat_path: str | Path) -> dict[str, Any]:
    import scipy.io as sio

    try:
        raw = sio.loadmat(
            str(mat_path),
            squeeze_me=False,
            struct_as_record=False,
            variable_names=["metadata", VALIDATION_RESULT_KEY, VALIDATION_PARAM_KEY],
        )
    except Exception:
        return {}
    return expand_public_mat_payload(raw)


def _validation_shell_surface_note(raw: Mapping[str, Any]) -> str:
    """Summarise shell surface extraction metadata without exposing extra fields."""
    if not _validation_is_shell(raw):
        return ""
    names = _validation_flat_strings(raw.get("validation_abaqus_surface_section_names")) or ["SNEG", "SPOS"]
    section_numbers = _validation_flat_ints(raw.get("validation_abaqus_surface_section_numbers"))
    source_indices = _validation_flat_ints(raw.get("validation_abaqus_source_surface_section_indices"))
    all_numbers = _validation_flat_ints(raw.get("validation_abaqus_all_section_numbers"))
    surface_count = _validation_flat_ints(raw.get("validation_abaqus_section_count"))
    all_count = _validation_flat_ints(raw.get("validation_abaqus_all_section_count"))
    count_text = ""
    if all_count:
        count_text = f" of {all_count[0]} ALLSECTIONPTS"
    elif all_numbers:
        count_text = f" of {len(all_numbers)} ALLSECTIONPTS"
    if section_numbers:
        pairs = [
            f"{name}=section {section_numbers[idx]}"
            for idx, name in enumerate(names)
            if idx < len(section_numbers)
        ]
        note = f"ABAQUS layer fields expose {', '.join(pairs)}"
    elif surface_count:
        note = f"ABAQUS layer fields expose {surface_count[0]} shell surfaces"
    else:
        note = f"ABAQUS layer fields expose {', '.join(names)} shell surfaces"
    if source_indices:
        idx_text = ", ".join(str(v) for v in source_indices)
        note = f"{note}; source section positions {idx_text}{count_text}"
    elif count_text:
        note = f"{note}{count_text}"
    return f"{note}. JAX fields use the same public SNEG/SPOS naming."


def _validation_display_spec(raw: Mapping[str, Any], semantic: str, arr: Any = None) -> dict[str, Any]:
    """Return display metadata for one validation semantic in the current MAT."""
    spec = dict(_VALIDATION_DISPLAY.get(semantic, {}))
    shell = _validation_is_shell(raw)
    try:
        if isinstance(arr, (tuple, list)) and arr and all(isinstance(v, (int, np.integer)) for v in arr):
            n_comp = int(arr[-1])
        else:
            n_comp = int(np.asarray(arr).shape[-1]) if arr is not None and np.asarray(arr).ndim > 0 else int(spec.get("n_components", 1))
    except Exception:
        n_comp = int(spec.get("n_components", 1))
    if shell and semantic == "S":
        spec.update({
            "label": "SGEN shell generalized stress history" if n_comp >= 6 else "SGEN shell membrane force history",
            "display_key": "SGEN",
            "symbol": r"\mathbf{S}_{gen}(t)",
            "formula": r"\mathbf{S}_{gen}(t)=[\mathbf{SF}(t),\mathbf{SM}(t)]",
            "description": "History of shell generalized stress resultants: SF membrane force and SM bending moment, compared with ABAQUS SF/SM.",
        })
    elif shell and semantic == "E":
        spec.update({
            "label": "EGEN shell generalized strain history",
            "display_key": "EGEN",
            "symbol": r"\mathbf{E}_{gen}(t)",
            "formula": r"\mathbf{E}_{gen}(t)=[\boldsymbol{\varepsilon}_{m}(t),\boldsymbol{\kappa}(t)]",
            "description": "History of shell generalized strain: membrane strain GE and curvature GK, compared with ABAQUS SE/SK.",
        })
    elif shell and semantic == "U":
        spec.update({
            "label": "U shell displacement/rotation history",
            "display_key": "U",
            "symbol": r"\mathbf{q}(t)",
            "formula": r"\mathbf{q}(t)=[U_1,U_2,U_3,UR_1,UR_2,UR_3]",
            "description": "Shell nodal translational and rotational degrees of freedom.",
        })
    elif shell and semantic == "NFORC":
        spec.update({
            "label": "NFORC shell generalized nodal force history",
            "display_key": "NFORC",
            "symbol": r"\mathbf{NFORC}(t)",
            "formula": r"\mathbf{NFORC}(t)=[F_1,F_2,F_3,M_1,M_2,M_3]",
            "description": "Shell nodal generalized force and moment compatible with ABAQUS NFORC1..6 output.",
        })
    elif shell and semantic.startswith("PE_layer") and n_comp <= 3:
        surface = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        spec.update({
            "formula": rf"\boldsymbol{{\varepsilon}}^{{p,S}}_{{{surface}}}(t)=\boldsymbol{{\varepsilon}}^{{p,S}}(z_{{{surface}}},t)=[PE_{{11}},PE_{{22}},PE_{{12}}]",
            "description": f"History of shell layer plastic strain components exposed for validation on the {surface} surface.",
        })
    if shell and (
        semantic.startswith(("S_layer", "E_layer", "PE_layer"))
        or semantic in {"PEEQ_Neg", "PEEQ_Pos", "PEEQ_Max"}
    ):
        note = _validation_shell_surface_note(raw)
        if note:
            desc = str(spec.get("description", "")).rstrip()
            spec["description"] = f"{desc} {note}".strip()
    return spec


def _validation_virtual_key(side: str, semantic: str) -> str:
    return f"{_VALIDATION_SIDE_PREFIX[side]}{semantic}"


def validation_counterpart_fields(selected_fields: Optional[Sequence[str]], *, target_side: str = "abaqus") -> list[str]:
    """Map selected validation fields from one side to the matching opposite side."""
    target_side = "abaqus" if str(target_side).lower() != "jax" else "jax"
    target_prefix = _VALIDATION_SIDE_PREFIX[target_side]
    out: list[str] = []
    for raw_key in selected_fields or []:
        text = str(raw_key).strip()
        semantic = ""
        for prefix in _VALIDATION_SIDE_PREFIX.values():
            if text.startswith(prefix):
                semantic = text[len(prefix):]
                break
        if not semantic:
            semantic = text
        key = f"{target_prefix}{semantic}"
        if key not in out:
            out.append(key)
    return out


def _as_validation_flag(raw: Mapping[str, Any]) -> bool:
    value = raw.get("validated_with_ODB")
    if value is None:
        return False
    try:
        return bool(int(np.asarray(value, dtype=np.int32).reshape(-1)[0]) == 1)
    except Exception:
        return bool(value)


def is_validation_mat(mat_path: str | Path) -> bool:
    """Lightweight check for ODB validation MATs.

    Two on-disk layouts must be recognised:

    * **legacy / flat** — ``validated_with_ODB`` lives at the top level
      (most existing ``Examples/*/validation/*.mat`` were saved this way).
    * **public / packed** — the flag was packed under
      ``VALIDATION_RESULT_KEY`` (``ValidationResult``) struct.

    The previous implementation only requested ``VALIDATION_RESULT_KEY``
    from ``loadmat``, which silently misclassified all legacy MATs as
    non-validation and broke the web auto-compare path.
    """
    import scipy.io as sio

    try:
        raw = sio.loadmat(
            str(mat_path),
            squeeze_me=False,
            struct_as_record=False,
            variable_names=[VALIDATION_RESULT_KEY, "validated_with_ODB"],
        )
    except Exception:
        return False
    raw = expand_public_mat_payload(raw)
    return _as_validation_flag(raw)


def _whos_has_validation(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> bool:
    names = _entry_names(entries)
    return VALIDATION_RESULT_KEY in names and any(
        name.startswith("abaqus_frame_")
        for name in names
    )


def _validation_entries_by_name(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict[str, tuple[int, ...]]:
    return {str(name): tuple(int(v) for v in shape) for name, shape, _ in entries}


def _validation_available_semantics_from_entries(entries: Sequence[tuple[str, tuple[int, ...], str]]) -> list[str]:
    shapes = _validation_entries_by_name(entries)
    out: list[str] = []

    def add(semantic: str, jax_keys: Sequence[str], aba_keys: Sequence[str]) -> None:
        if any(key in shapes for key in jax_keys) and any(key in shapes for key in aba_keys):
            out.append(semantic)

    def has_pair(jax_keys: Sequence[str], aba_keys: Sequence[str]) -> bool:
        return any(key in shapes for key in jax_keys) and any(key in shapes for key in aba_keys)

    add("U", ("solid_u_nodal", "shell_u_nodal", "u", "elastic_u", "frame_u"), ("abaqus_u_nodal", "abaqus_frame_u_nodal"))
    add("NFORC", ("solid_nforc_nodal", "shell_gen_internal_force", "internal_force", "frame_nforc", "frame_internal_force"), ("abaqus_nforc", "abaqus_frame_nforc"))
    add("S", ("gauss_stress", "frame_gauss_stress"), ("abaqus_gauss_stress", "abaqus_frame_gauss_stress", "abaqus_gen_section_force"))
    add("E", ("gauss_strain", "frame_gauss_strain"), ("abaqus_gauss_strain", "abaqus_frame_gauss_strain", "abaqus_gen_strain"))
    add(
        "S_layer_SNEG",
        ("validation_jax_shell_layer_s", "frame_shell_layer_stress_SNEG", "frame_shell_layer_stress", "shell_layer_stress"),
        ("validation_abaqus_shell_layer_s", "abaqus_frame_shell_layer_stress_SNEG", "abaqus_layer_stress"),
    )
    add(
        "S_layer_SPOS",
        ("validation_jax_shell_layer_s", "frame_shell_layer_stress_SPOS", "frame_shell_layer_stress", "shell_layer_stress"),
        ("validation_abaqus_shell_layer_s", "abaqus_frame_shell_layer_stress_SPOS", "abaqus_layer_stress"),
    )
    add(
        "E_layer_SNEG",
        ("validation_jax_shell_layer_e", "frame_shell_layer_strain_SNEG", "frame_shell_layer_strain", "shell_layer_strain"),
        ("validation_abaqus_shell_layer_e", "abaqus_frame_shell_layer_strain_SNEG", "abaqus_layer_strain"),
    )
    add(
        "E_layer_SPOS",
        ("validation_jax_shell_layer_e", "frame_shell_layer_strain_SPOS", "frame_shell_layer_strain", "shell_layer_strain"),
        ("validation_abaqus_shell_layer_e", "abaqus_frame_shell_layer_strain_SPOS", "abaqus_layer_strain"),
    )
    add(
        "PE_layer_SNEG",
        ("validation_jax_shell_layer_pe", "frame_shell_layer_plastic_strain_SNEG", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain"),
        ("validation_abaqus_shell_layer_pe", "abaqus_frame_shell_layer_plastic_strain_SNEG", "abaqus_layer_plastic_strain"),
    )
    add(
        "PE_layer_SPOS",
        ("validation_jax_shell_layer_pe", "frame_shell_layer_plastic_strain_SPOS", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain"),
        ("validation_abaqus_shell_layer_pe", "abaqus_frame_shell_layer_plastic_strain_SPOS", "abaqus_layer_plastic_strain"),
    )
    shell_peeq_pair = has_pair(
        ("validation_jax_shell_layer_peeq", "frame_shell_layer_peeq_SNEG", "frame_shell_layer_peeq_SPOS", "frame_shell_layer_peeq"),
        ("validation_abaqus_shell_layer_peeq", "abaqus_frame_shell_layer_peeq_SNEG", "abaqus_frame_shell_layer_peeq_SPOS"),
    )
    solid_plastic_strain_pair = has_pair(
        ("frame_gauss_plast_strain", "frame_gauss_plastic_strain", "gauss_plastic_strain"),
        ("abaqus_frame_gauss_plast_strain", "abaqus_frame_gauss_plastic_strain", "abaqus_gauss_plastic_strain"),
    )
    if not shell_peeq_pair:
        add(
            "PEEQ",
            ("validation_jax_shell_layer_peeq", "gauss_peeq", "gauss_eqps", "frame_gauss_peeq", "frame_shell_layer_peeq"),
            ("validation_abaqus_shell_layer_peeq", "abaqus_gauss_peeq", "abaqus_frame_gauss_peeq"),
        )
    if solid_plastic_strain_pair:
        out.append("PE")
    add(
        "PEEQ_Neg",
        ("validation_jax_shell_layer_peeq", "frame_shell_layer_peeq_SNEG", "frame_shell_layer_peeq"),
        ("validation_abaqus_shell_layer_peeq", "abaqus_frame_shell_layer_peeq_SNEG"),
    )
    add(
        "PEEQ_Pos",
        ("validation_jax_shell_layer_peeq", "frame_shell_layer_peeq_SPOS", "frame_shell_layer_peeq"),
        ("validation_abaqus_shell_layer_peeq", "abaqus_frame_shell_layer_peeq_SPOS"),
    )
    if shell_peeq_pair:
        add(
            "PEEQ_Max",
            ("validation_jax_shell_layer_peeq", "frame_gauss_peeq", "frame_shell_layer_peeq"),
            ("validation_abaqus_shell_layer_peeq", "abaqus_gauss_peeq", "abaqus_frame_gauss_peeq"),
        )
    return out


def _validation_inspect_shape_for_source(
    shapes: Mapping[str, tuple[int, ...]],
    semantic: str,
    source_key: str,
) -> tuple[list[int], bool]:
    shape = list(shapes.get(source_key, ()))
    is_frame = (
        source_key.startswith("frame_")
        or source_key.startswith("validation_jax_")
        or source_key.startswith("validation_abaqus_")
    )
    if semantic == "U" and source_key == "frame_u" and len(shape) == 2:
        if "abaqus_frame_u_nodal" in shapes and len(shapes["abaqus_frame_u_nodal"]) >= 3:
            return list(shapes["abaqus_frame_u_nodal"]), True
        if "abaqus_u_nodal" in shapes and len(shapes["abaqus_u_nodal"]) >= 2:
            return [shape[0], *list(shapes["abaqus_u_nodal"])], True
    if semantic == "NFORC" and source_key in {"frame_nforc", "frame_internal_force"} and len(shape) == 2:
        if "abaqus_frame_nforc" in shapes and len(shapes["abaqus_frame_nforc"]) >= 3:
            return list(shapes["abaqus_frame_nforc"]), True
        if "abaqus_nforc" in shapes and len(shapes["abaqus_nforc"]) >= 2:
            return [shape[0], *list(shapes["abaqus_nforc"])], True
    if semantic == "U" and len(shape) == 1 and "abaqus_u_nodal" in shapes:
        return list(shapes["abaqus_u_nodal"]), is_frame
    if semantic == "NFORC" and len(shape) == 1 and "abaqus_nforc" in shapes:
        return list(shapes["abaqus_nforc"]), is_frame
    return shape, is_frame


def inspect_validation_fields(mat_path: str | Path, entries: Sequence[tuple[str, tuple[int, ...], str]]) -> dict:
    """Return the JAX-side field catalog for an embedded ODB validation MAT."""
    shapes = _validation_entries_by_name(entries)
    display_raw: dict[str, Any] = dict(shapes)
    display_raw.update(_load_validation_display_metadata(mat_path))
    fields: list[dict[str, Any]] = []
    for semantic in _validation_available_semantics_from_entries(entries):
        key = _validation_virtual_key("jax", semantic)
        shape = []
        is_frame = False
        for source_key in (
            "frame_u", "solid_u_nodal", "shell_u_nodal", "u", "elastic_u",
            "frame_gauss_stress", "gauss_stress", "frame_gauss_strain", "gauss_strain",
            "validation_jax_shell_layer_s", "frame_shell_layer_stress_SNEG", "frame_shell_layer_stress_SPOS", "frame_shell_layer_stress", "shell_layer_stress",
            "validation_jax_shell_layer_e", "frame_shell_layer_strain_SNEG", "frame_shell_layer_strain_SPOS", "frame_shell_layer_strain", "shell_layer_strain",
            "validation_jax_shell_layer_pe", "frame_shell_layer_plastic_strain_SNEG", "frame_shell_layer_plastic_strain_SPOS", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain",
            "frame_gauss_plast_strain", "frame_gauss_plastic_strain", "gauss_plastic_strain",
            "frame_nforc", "frame_internal_force", "solid_nforc_nodal",
            "shell_gen_internal_force", "internal_force",
            "validation_jax_shell_layer_peeq", "frame_gauss_peeq", "frame_shell_layer_peeq_SNEG", "frame_shell_layer_peeq_SPOS", "frame_shell_layer_peeq", "gauss_peeq", "gauss_eqps",
        ):
            if source_key in shapes:
                if semantic == "U" and source_key not in {"solid_u_nodal", "shell_u_nodal", "u", "elastic_u", "frame_u"}:
                    continue
                if semantic == "NFORC" and source_key not in {"solid_nforc_nodal", "shell_gen_internal_force", "internal_force", "frame_nforc", "frame_internal_force"}:
                    continue
                if semantic == "S" and source_key not in {"gauss_stress", "frame_gauss_stress"}:
                    continue
                if semantic == "E" and source_key not in {"gauss_strain", "frame_gauss_strain"}:
                    continue
                if semantic.startswith("S_layer") and source_key not in {"validation_jax_shell_layer_s", "frame_shell_layer_stress_SNEG", "frame_shell_layer_stress_SPOS", "frame_shell_layer_stress", "shell_layer_stress"}:
                    continue
                if semantic.startswith("E_layer") and source_key not in {"validation_jax_shell_layer_e", "frame_shell_layer_strain_SNEG", "frame_shell_layer_strain_SPOS", "frame_shell_layer_strain", "shell_layer_strain"}:
                    continue
                if semantic.startswith("PE_layer") and source_key not in {"validation_jax_shell_layer_pe", "frame_shell_layer_plastic_strain_SNEG", "frame_shell_layer_plastic_strain_SPOS", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain"}:
                    continue
                if semantic == "PE" and source_key not in {"frame_gauss_plast_strain", "frame_gauss_plastic_strain", "gauss_plastic_strain"}:
                    continue
                if semantic.startswith("PEEQ") and source_key not in {"validation_jax_shell_layer_peeq", "gauss_peeq", "gauss_eqps", "frame_gauss_peeq", "frame_shell_layer_peeq_SNEG", "frame_shell_layer_peeq_SPOS", "frame_shell_layer_peeq"}:
                    continue
                shape, is_frame = _validation_inspect_shape_for_source(shapes, semantic, source_key)
                break
        spec = _validation_display_spec(display_raw, semantic, shape)
        n_components = 1 if spec["tensor_kind"] == "scalar" else (int(shape[-1]) if shape else 1)
        if semantic.endswith(("SNEG", "SPOS")) and len(shape) >= 5:
            shape = [shape[0], shape[1], shape[2], shape[-1]]
        if semantic in {"PEEQ", "PEEQ_Max", "PEEQ_Neg", "PEEQ_Pos"} and len(shape) >= 4:
            if len(shape) >= 5 and int(shape[-1]) == 1:
                shape = [shape[0], shape[1], shape[2]]
            elif int(shape[-1]) in {1, 2, 3, 5}:
                shape = [shape[0], shape[1], shape[2]]
            n_components = 1
        fields.append({
            "key": key,
            "label": f"JAX {spec['label']}",
            "display_key": str(spec.get("display_key", semantic)),
            "semantic": semantic,
            "shape": shape,
            "mat_class": "validation",
            "location": spec["location"],
            "frames": "frames" if is_frame else "single",
            "n_components": int(n_components),
            "family": spec["family"],
            "tensor_kind": spec["tensor_kind"],
            "default_selected": semantic in {"U", "S", "E", "NFORC", "PEEQ", "PE"},
            "visualizable": True,
            "validation_side": "jax",
            "validation_pair_key": _validation_virtual_key("abaqus", semantic),
            "symbol": str(spec.get("symbol", "")),
            "formula": str(spec.get("formula", "")),
            "description": str(spec.get("description", "")),
        })
    fields.sort(key=_field_sort_key)
    return {
        "source": str(mat_path),
        "fields": fields,
        "validation_compare": True,
        "validation_compare_mode": "embedded_odb",
    }


def _validation_required_raw_keys(selected_fields: Optional[set[str]], side: Optional[str]) -> set[str]:
    selected = set(selected_fields or [])
    semantics: set[str] = set()
    for key in selected:
        for prefix in _VALIDATION_SIDE_PREFIX.values():
            if key.startswith(prefix):
                semantics.add(key[len(prefix):])
                break
    if not semantics:
        semantics.update(_VALIDATION_DISPLAY)
    required = {
        "InpData",
        FRAME_OUTPUTS_KEY,
        SHELL_PARAM_KEY,
        SOLID_PARAM_KEY,
        VALIDATION_RESULT_KEY,
        VALIDATION_PARAM_KEY,
        VIZ_MANIFEST_KEY,
        "metadata",
        "abaqus_node_coords",
        "abaqus_node_labels",
        "abaqus_elem_conn",
        "abaqus_elem_types",
        "abaqus_elem_labels",
        "solid_elem_types",
        "frame_time",
        "time_grid",
        "frame_load_scale",
    }
    for semantic in semantics:
        if semantic == "U":
            required.update({"solid_u_nodal", "shell_u_nodal", "u", "elastic_u", "frame_u", "abaqus_u_nodal", "abaqus_frame_u_nodal"})
        elif semantic == "NFORC":
            required.update({"solid_nforc_nodal", "shell_gen_internal_force", "internal_force", "frame_nforc", "frame_internal_force", "abaqus_nforc", "abaqus_frame_nforc"})
        elif semantic == "S":
            required.update({"gauss_stress", "frame_gauss_stress", "abaqus_gauss_stress", "abaqus_frame_gauss_stress", "abaqus_gen_section_force", "abaqus_gen_section_moment"})
        elif semantic == "E":
            required.update({"gauss_strain", "frame_gauss_strain", "abaqus_gauss_strain", "abaqus_frame_gauss_strain", "abaqus_gen_strain", "abaqus_gen_curvature"})
        elif semantic.startswith("S_layer"):
            required.update({"validation_jax_shell_layer_s", "frame_shell_layer_stress_SNEG", "frame_shell_layer_stress_SPOS", "frame_shell_layer_stress", "shell_layer_stress", "validation_abaqus_shell_layer_s", "abaqus_frame_shell_layer_stress_SNEG", "abaqus_frame_shell_layer_stress_SPOS", "abaqus_layer_stress"})
        elif semantic.startswith("E_layer"):
            required.update({"validation_jax_shell_layer_e", "frame_shell_layer_strain_SNEG", "frame_shell_layer_strain_SPOS", "frame_shell_layer_strain", "shell_layer_strain", "validation_abaqus_shell_layer_e", "abaqus_frame_shell_layer_strain_SNEG", "abaqus_frame_shell_layer_strain_SPOS", "abaqus_layer_strain"})
        elif semantic.startswith("PE_layer"):
            required.update({"validation_jax_shell_layer_pe", "frame_shell_layer_plastic_strain_SNEG", "frame_shell_layer_plastic_strain_SPOS", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain", "validation_abaqus_shell_layer_pe", "abaqus_frame_shell_layer_plastic_strain_SNEG", "abaqus_frame_shell_layer_plastic_strain_SPOS", "abaqus_layer_plastic_strain"})
        elif semantic == "PE":
            required.update({
                "frame_gauss_plast_strain",
                "frame_gauss_plastic_strain",
                "gauss_plastic_strain",
                "abaqus_frame_gauss_plast_strain",
                "abaqus_frame_gauss_plastic_strain",
                "abaqus_gauss_plastic_strain",
            })
        elif semantic.startswith("PEEQ"):
            required.update({
                "validation_jax_shell_layer_peeq",
                "validation_abaqus_shell_layer_peeq",
                "gauss_peeq",
                "gauss_eqps",
                "frame_gauss_peeq",
                "frame_shell_layer_peeq_SNEG",
                "frame_shell_layer_peeq_SPOS",
                "frame_shell_layer_peeq",
                "abaqus_gauss_peeq",
                "abaqus_frame_gauss_peeq",
                "abaqus_frame_shell_layer_peeq_SNEG",
                "abaqus_frame_shell_layer_peeq_SPOS",
            })
    return required


def _coord_key_for_validation(coord: np.ndarray, decimals: int = 8) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(coord, dtype=np.float64).reshape(-1), decimals=decimals).tolist())


def _validation_reorder_by_coords(source_coords: Optional[np.ndarray], target_coords: Optional[np.ndarray], values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if source_coords is None or target_coords is None:
        return vals
    src = np.asarray(source_coords, dtype=np.float64)
    tgt = np.asarray(target_coords, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2:
        return vals
    has_frame_axis = vals.ndim >= 3 and vals.shape[1] == src.shape[0]
    if vals.shape[0] != src.shape[0] and not has_frame_axis:
        return vals
    dim = min(src.shape[1], tgt.shape[1])
    src_cmp = src[:, :dim]
    lookup = {_coord_key_for_validation(coord): idx for idx, coord in enumerate(src_cmp)}
    order: list[int] = []
    for coord in tgt[:, :dim]:
        idx = lookup.get(_coord_key_for_validation(coord))
        if idx is None:
            d = np.linalg.norm(src_cmp - coord[None, :], axis=1)
            idx = int(np.argmin(d))
            if float(d[idx]) > 1.0e-6:
                return vals
        order.append(idx)
    order_arr = np.asarray(order, dtype=np.int64)
    if has_frame_axis:
        return vals[:, order_arr, ...]
    return vals[order_arr]


def _validation_cell_connectivity(vd: VizData) -> Optional[np.ndarray]:
    if vd.cells is None or vd.cell_types is None:
        return None
    rows: list[np.ndarray] = []
    cells = np.asarray(vd.cells, dtype=np.int64).reshape(-1)
    pos = 0
    for _ in range(int(np.asarray(vd.cell_types).size)):
        if pos >= cells.size:
            break
        npe = int(cells[pos])
        rows.append(cells[pos + 1: pos + 1 + npe])
        pos += 1 + npe
    if not rows:
        return None
    width = max(len(row) for row in rows)
    out = -np.ones((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _validation_element_centroids(points: Optional[np.ndarray], conn: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if points is None or conn is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    rows = []
    for row in np.asarray(conn, dtype=np.int64):
        ids = row[row >= 0]
        if ids.size == 0 or np.max(ids) >= pts.shape[0]:
            return None
        rows.append(np.mean(pts[ids], axis=0))
    return np.asarray(rows, dtype=np.float64)


def _validation_abaqus_connectivity(raw: Mapping[str, Any]) -> Optional[np.ndarray]:
    conn = _as_float_array(raw.get("abaqus_elem_conn"), squeeze=True)
    if conn is None:
        return None
    arr = np.asarray(conn, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    labels = _as_float_array(raw.get("abaqus_node_labels"), squeeze=True)
    if labels is not None:
        label_to_idx = {int(label): idx for idx, label in enumerate(np.asarray(labels, dtype=np.int64).reshape(-1))}
        mapped = np.zeros_like(arr, dtype=np.int64)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                mapped[i, j] = label_to_idx.get(int(arr[i, j]), int(arr[i, j]) - 1)
        arr = mapped
    elif arr.min(initial=0) >= 1:
        arr = arr - 1
    return arr


def _validation_element_order_to_jax(raw: Mapping[str, Any], vd: VizData) -> Optional[np.ndarray]:
    target = _validation_element_centroids(vd.points, _validation_cell_connectivity(vd))
    aba_points = _as_float_array(raw.get("abaqus_node_coords"), squeeze=True)
    source = _validation_element_centroids(aba_points, _validation_abaqus_connectivity(raw))
    if source is None or target is None or source.shape[0] != target.shape[0]:
        return None
    dim = min(source.shape[1], target.shape[1])
    lookup = {_coord_key_for_validation(coord): idx for idx, coord in enumerate(source[:, :dim])}
    order: list[int] = []
    for coord in target[:, :dim]:
        idx = lookup.get(_coord_key_for_validation(coord))
        if idx is None:
            d = np.linalg.norm(source[:, :dim] - coord[None, :], axis=1)
            idx = int(np.argmin(d))
            if float(d[idx]) > 1.0e-6:
                return None
        order.append(idx)
    return np.asarray(order, dtype=np.int64)


def _validation_reorder_elements_to_jax(raw: Mapping[str, Any], vd: VizData, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = _validation_element_order_to_jax(raw, vd)
    if arr.ndim == 0:
        return arr
    ngpe = np.asarray(vd.n_gauss_per_elem, dtype=np.int32).reshape(-1) if vd.n_gauss_per_elem is not None else None
    if order is None:
        if ngpe is not None and np.unique(ngpe).size == 1:
            n_elem = int(ngpe.size)
            n_gp = int(ngpe[0])
            n_total = int(np.sum(ngpe))
            if arr.shape[0] == n_total:
                return arr.reshape((n_elem, n_gp) + arr.shape[1:])
            if arr.ndim >= 2 and arr.shape[1] == n_total:
                return arr.reshape((arr.shape[0], n_elem, n_gp) + arr.shape[2:])
        return arr
    n_elem = int(order.shape[0])
    if arr.shape[0] == n_elem:
        return arr[order]
    if arr.ndim >= 3 and arr.shape[1] == n_elem:
        return arr[:, order, ...]
    if ngpe is not None and ngpe.size == n_elem and np.unique(ngpe).size == 1 and arr.shape[0] == int(np.sum(ngpe)):
        n_gp = int(ngpe[0])
        reshaped = arr.reshape((n_elem, n_gp) + arr.shape[1:])
        return reshaped[order]
    if (
        ngpe is not None
        and ngpe.size == n_elem
        and np.unique(ngpe).size == 1
        and arr.ndim >= 2
        and arr.shape[1] == int(np.sum(ngpe))
    ):
        n_gp = int(ngpe[0])
        reshaped = arr.reshape((arr.shape[0], n_elem, n_gp) + arr.shape[2:])
        return reshaped[:, order, ...]
    return arr


def _validation_take_component_count(arr: np.ndarray, n_comp: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == n_comp:
        return values
    if values.shape[-1] > n_comp:
        return values[..., :n_comp]
    pad = [(0, 0)] * values.ndim
    pad[-1] = (0, n_comp - values.shape[-1])
    return np.pad(values, pad, mode="constant", constant_values=0.0)


def _validation_plane_components(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim > 0 and values.shape[-1] >= 4:
        return np.stack([values[..., 0], values[..., 1], values[..., 3]], axis=-1)
    return values[..., :3]


def _validation_section_surface(arr: np.ndarray, surface: str) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim >= 3:
        has_component_axis = values.ndim >= 4 and values.shape[-1] in {1, 3, 4, 6}
        axis = values.ndim - 2 if has_component_axis else values.ndim - 1
        idx = 0 if surface.upper() == "SNEG" else values.shape[axis] - 1
        values = np.take(values, idx, axis=axis)
    return values


def _validation_direct_layer_key(side: str, semantic: str) -> Optional[str]:
    prefix = _VALIDATION_SIDE_PREFIX.get(side)
    if prefix is None:
        return None
    if semantic.startswith("S_layer"):
        return f"{prefix}shell_layer_s"
    if semantic.startswith("E_layer"):
        return f"{prefix}shell_layer_e"
    if semantic.startswith("PE_layer"):
        return f"{prefix}shell_layer_pe"
    if semantic.startswith("PEEQ"):
        return f"{prefix}shell_layer_peeq"
    return None


def _validation_direct_field(raw: Mapping[str, Any], side: str, semantic: str) -> Optional[np.ndarray]:
    key = _validation_direct_layer_key(side, semantic)
    if key is None:
        return None
    arr = _as_float_array(raw.get(key), squeeze=True)
    if arr is None:
        return None
    values = np.asarray(arr, dtype=np.float64)
    if semantic.endswith("SNEG") or semantic == "PEEQ_Neg":
        values = _validation_section_surface(values, "SNEG")
    elif semantic.endswith("SPOS") or semantic == "PEEQ_Pos":
        values = _validation_section_surface(values, "SPOS")
    elif semantic in {"PEEQ", "PEEQ_Max"} and values.ndim >= 3:
        has_component_axis = values.ndim >= 4 and values.shape[-1] in {1, 3, 4, 6}
        axis = values.ndim - 2 if has_component_axis else values.ndim - 1
        values = np.max(values, axis=axis)
    return values


def _validation_reshape_flat_nodal(raw: Mapping[str, Any], arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim not in {1, 2}:
        return values
    n_nodes = 0
    trailing_dim = 0
    try:
        inp = _unwrap_scalar(raw.get("InpData")) if hasattr(raw, "get") else None
        points = inp.get("points") if isinstance(inp, dict) else getattr(inp, "points", None)
        if points is not None:
            n_nodes = int(np.asarray(points).shape[0])
    except Exception:
        n_nodes = 0
    if n_nodes <= 0:
        try:
            aba_u = None
            if hasattr(raw, "get"):
                aba_u = raw.get("abaqus_frame_u_nodal")
                if aba_u is None:
                    aba_u = raw.get("abaqus_u_nodal")
            if aba_u is not None:
                aba_shape = np.asarray(aba_u).shape
                if len(aba_shape) >= 3:
                    n_nodes = int(aba_shape[1])
                    trailing_dim = int(aba_shape[2])
                elif len(aba_shape) >= 2:
                    n_nodes = int(aba_shape[0])
                    trailing_dim = int(aba_shape[1])
        except Exception:
            n_nodes = 0
    if n_nodes > 0 and trailing_dim <= 0:
        flat_width = int(values.shape[-1]) if values.ndim == 2 else int(values.size)
        trailing_dim = flat_width // n_nodes if flat_width % n_nodes == 0 else 0
    if n_nodes > 0 and trailing_dim > 0 and values.ndim == 1 and values.size == n_nodes * trailing_dim:
        return values.reshape(n_nodes, values.size // n_nodes)
    if n_nodes > 0 and trailing_dim > 0 and values.ndim == 2 and values.shape[1] == n_nodes * trailing_dim:
        return values.reshape(values.shape[0], n_nodes, trailing_dim)
    return values


def _validation_shell_gen_force(sf: np.ndarray) -> np.ndarray:
    return _validation_plane_components(sf)


def _validation_shell_gen_moment(sm: np.ndarray) -> np.ndarray:
    values = np.asarray(sm, dtype=np.float64)
    if values.shape[-1] >= 3:
        return np.stack([values[..., 1], values[..., 0], values[..., 2]], axis=-1)
    return values


def _validation_raw_jax_field(raw: Mapping[str, Any], semantic: str) -> Optional[np.ndarray]:
    direct = _validation_direct_field(raw, "jax", semantic)
    if direct is not None:
        return direct
    candidates: tuple[str, ...]
    if semantic == "U":
        candidates = ("frame_u", "solid_u_nodal", "shell_u_nodal", "u", "elastic_u")
    elif semantic == "NFORC":
        candidates = ("frame_nforc", "frame_internal_force", "solid_nforc_nodal", "shell_gen_internal_force", "internal_force")
    elif semantic == "S":
        candidates = ("frame_gauss_stress", "gauss_stress")
    elif semantic == "E":
        candidates = ("frame_gauss_strain", "gauss_strain")
    elif semantic.startswith("S_layer"):
        surface_key = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        candidates = (f"frame_shell_layer_stress_{surface_key}", "frame_shell_layer_stress", "shell_layer_stress")
    elif semantic.startswith("E_layer"):
        surface_key = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        candidates = (f"frame_shell_layer_strain_{surface_key}", "frame_shell_layer_strain", "shell_layer_strain")
    elif semantic.startswith("PE_layer"):
        surface_key = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        candidates = (f"frame_shell_layer_plastic_strain_{surface_key}", "frame_shell_layer_plastic_strain", "shell_layer_plastic_strain")
    elif semantic == "PE":
        candidates = ("frame_gauss_plast_strain", "frame_gauss_plastic_strain", "gauss_plastic_strain")
    elif semantic == "PEEQ_Neg":
        candidates = ("frame_shell_layer_peeq_SNEG", "frame_shell_layer_peeq")
    elif semantic == "PEEQ_Pos":
        candidates = ("frame_shell_layer_peeq_SPOS", "frame_shell_layer_peeq")
    elif semantic == "PEEQ":
        candidates = ("frame_gauss_peeq", "frame_shell_layer_peeq", "gauss_peeq", "gauss_eqps")
    elif semantic == "PEEQ_Max":
        if "frame_shell_layer_peeq_SNEG" in raw and "frame_shell_layer_peeq_SPOS" in raw:
            neg = np.asarray(raw["frame_shell_layer_peeq_SNEG"], dtype=np.float64)
            pos = np.asarray(raw["frame_shell_layer_peeq_SPOS"], dtype=np.float64)
            return np.maximum(neg, pos)
        candidates = ("frame_shell_layer_peeq",)
    else:
        candidates = ()
    for key in candidates:
        if key in raw:
            arr = np.asarray(raw[key], dtype=np.float64)
            if semantic in {"U", "NFORC"}:
                arr = _validation_reshape_flat_nodal(raw, arr)
            if (semantic.endswith("SNEG") and not key.endswith("_SNEG")) or (semantic == "PEEQ_Neg" and not key.endswith("_SNEG")):
                arr = _validation_section_surface(arr, "SNEG")
            elif (semantic.endswith("SPOS") and not key.endswith("_SPOS")) or (semantic == "PEEQ_Pos" and not key.endswith("_SPOS")):
                arr = _validation_section_surface(arr, "SPOS")
            elif semantic in {"PEEQ", "PEEQ_Max"} and key == "frame_shell_layer_peeq":
                has_component_axis = arr.ndim >= 4 and arr.shape[-1] in {1, 3, 4, 6}
                axis = arr.ndim - 2 if has_component_axis else arr.ndim - 1
                arr = np.max(arr, axis=axis)
            return arr
    return None


def _validation_raw_abaqus_field(raw: Mapping[str, Any], semantic: str, vd: VizData) -> Optional[np.ndarray]:
    arr: Optional[np.ndarray] = None
    direct = _validation_direct_field(raw, "abaqus", semantic)
    if direct is not None:
        arr = direct
    elif semantic == "U":
        arr = _as_float_array(raw.get("abaqus_frame_u_nodal"), squeeze=True)
        if arr is None:
            arr = _as_float_array(raw.get("abaqus_u_nodal"), squeeze=True)
        if arr is not None:
            arr = _validation_reorder_by_coords(_as_float_array(raw.get("abaqus_node_coords"), squeeze=True), vd.points, arr)
    elif semantic == "NFORC":
        arr = _as_float_array(raw.get("abaqus_frame_nforc"), squeeze=True)
        if arr is None:
            arr = _as_float_array(raw.get("abaqus_nforc"), squeeze=True)
        if arr is not None:
            arr = _validation_reorder_by_coords(_as_float_array(raw.get("abaqus_node_coords"), squeeze=True), vd.points, arr)
    elif semantic == "S":
        direct = _as_float_array(raw.get("abaqus_frame_gauss_stress"), squeeze=True)
        if direct is None:
            direct = _as_float_array(raw.get("abaqus_gauss_stress"), squeeze=True)
        if direct is not None:
            arr = direct
        else:
            sf = _as_float_array(raw.get("abaqus_gen_section_force"), squeeze=True)
            sm = _as_float_array(raw.get("abaqus_gen_section_moment"), squeeze=True)
            if sf is not None and sm is not None:
                arr = np.concatenate([_validation_shell_gen_force(sf), _validation_shell_gen_moment(sm)], axis=-1)
    elif semantic == "E":
        direct = _as_float_array(raw.get("abaqus_frame_gauss_strain"), squeeze=True)
        if direct is None:
            direct = _as_float_array(raw.get("abaqus_gauss_strain"), squeeze=True)
        if direct is not None:
            arr = direct
        else:
            se = _as_float_array(raw.get("abaqus_gen_strain"), squeeze=True)
            sk = _as_float_array(raw.get("abaqus_gen_curvature"), squeeze=True)
            if se is not None and sk is not None:
                arr = np.concatenate([_validation_plane_components(se), _validation_shell_gen_moment(sk)], axis=-1)
    elif semantic.startswith("S_layer"):
        surface = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        direct = _as_float_array(raw.get(f"abaqus_frame_shell_layer_stress_{surface}"), squeeze=True)
        if direct is not None:
            arr = _validation_plane_components(direct)
        else:
            direct = _as_float_array(raw.get("abaqus_layer_stress"), squeeze=True)
            if direct is not None:
                arr = _validation_section_surface(_validation_plane_components(direct), surface)
    elif semantic.startswith("E_layer"):
        surface = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        direct = _as_float_array(raw.get(f"abaqus_frame_shell_layer_strain_{surface}"), squeeze=True)
        if direct is not None:
            arr = _validation_plane_components(direct)
        else:
            direct = _as_float_array(raw.get("abaqus_layer_strain"), squeeze=True)
            if direct is not None:
                arr = _validation_section_surface(_validation_plane_components(direct), surface)
    elif semantic.startswith("PE_layer"):
        surface = "SNEG" if semantic.endswith("SNEG") else "SPOS"
        direct = _as_float_array(raw.get(f"abaqus_frame_shell_layer_plastic_strain_{surface}"), squeeze=True)
        if direct is not None:
            arr = _validation_plane_components(direct)
        else:
            direct = _as_float_array(raw.get("abaqus_layer_plastic_strain"), squeeze=True)
            if direct is not None:
                arr = _validation_section_surface(_validation_plane_components(direct), surface)
    elif semantic == "PE":
        arr = _as_float_array(raw.get("abaqus_frame_gauss_plast_strain"), squeeze=True)
        if arr is None:
            arr = _as_float_array(raw.get("abaqus_frame_gauss_plastic_strain"), squeeze=True)
        if arr is None:
            arr = _as_float_array(raw.get("abaqus_gauss_plastic_strain"), squeeze=True)
    elif semantic == "PEEQ_Neg":
        arr = _as_float_array(raw.get("abaqus_frame_shell_layer_peeq_SNEG"), squeeze=True)
    elif semantic == "PEEQ_Pos":
        arr = _as_float_array(raw.get("abaqus_frame_shell_layer_peeq_SPOS"), squeeze=True)
    elif semantic in {"PEEQ", "PEEQ_Max"}:
        if semantic == "PEEQ_Max":
            neg = _as_float_array(raw.get("abaqus_frame_shell_layer_peeq_SNEG"), squeeze=True)
            pos = _as_float_array(raw.get("abaqus_frame_shell_layer_peeq_SPOS"), squeeze=True)
            if neg is not None and pos is not None:
                arr = np.maximum(np.asarray(neg, dtype=np.float64), np.asarray(pos, dtype=np.float64))
            elif "validation_abaqus_shell_layer_peeq" in raw or "abaqus_frame_shell_layer_peeq_SNEG" in raw or "abaqus_frame_shell_layer_peeq_SPOS" in raw:
                arr = _as_float_array(raw.get("abaqus_frame_gauss_peeq"), squeeze=True)
                if arr is None:
                    arr = _as_float_array(raw.get("abaqus_gauss_peeq"), squeeze=True)
        else:
            arr = _as_float_array(raw.get("abaqus_frame_gauss_peeq"), squeeze=True)
            if arr is None:
                arr = _as_float_array(raw.get("abaqus_gauss_peeq"), squeeze=True)
    if arr is None:
        return None
    if semantic not in {"U", "NFORC"}:
        arr = _validation_reorder_elements_to_jax(raw, vd, arr)
    return np.asarray(arr, dtype=np.float64)


def _validation_frame_count(arr: np.ndarray, vd: VizData, location: str) -> int:
    values = np.asarray(arr)
    if values.ndim < 2:
        return 1
    loc = str(location)
    if loc == "node":
        n_nodes = int(np.asarray(vd.points).shape[0]) if vd.points is not None else 0
        if n_nodes > 0 and values.ndim >= 3 and values.shape[1] == n_nodes:
            return int(values.shape[0])
        return 1
    if loc in {"cell", "gauss"}:
        n_elem = 0
        if vd.n_gauss_per_elem is not None:
            n_elem = int(np.asarray(vd.n_gauss_per_elem).size)
        elif vd.cell_types is not None:
            n_elem = int(np.asarray(vd.cell_types).size)
        if n_elem > 0 and values.ndim >= 3 and values.shape[1] == n_elem:
            return int(values.shape[0])
        if loc == "gauss" and vd.n_gauss_per_elem is not None:
            n_gauss = int(np.sum(np.asarray(vd.n_gauss_per_elem, dtype=np.int32).reshape(-1)))
            if n_gauss > 0 and values.ndim >= 2 and values.shape[1] == n_gauss:
                return int(values.shape[0])
    return 1


def _validation_semantic_from_key(key: str) -> str:
    text = str(key)
    for prefix in _VALIDATION_SIDE_PREFIX.values():
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _register_validation_compare_fields(
    raw: Mapping[str, Any],
    vd: VizData,
    selected_fields: Optional[set[str]],
    *,
    side: str,
) -> None:
    if side not in _VALIDATION_SIDE_PREFIX:
        return
    semantics = [_validation_semantic_from_key(key) for key in sorted(selected_fields or [])]
    if not semantics:
        semantics = list(_VALIDATION_DISPLAY)
    seen: set[str] = set()
    for semantic in semantics:
        if semantic in seen or semantic not in _VALIDATION_DISPLAY:
            continue
        seen.add(semantic)
        arr = (
            _validation_raw_jax_field(raw, semantic)
            if side == "jax"
            else _validation_raw_abaqus_field(raw, semantic, vd)
        )
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        if semantic not in {"U", "NFORC"}:
            arr = _validation_reorder_elements_to_jax(raw, vd, arr)
        if side == "abaqus" and semantic != "PEEQ":
            jax_arr = _validation_raw_jax_field(raw, semantic)
            if jax_arr is not None and np.asarray(jax_arr).ndim > 0:
                arr = _validation_take_component_count(arr, int(np.asarray(jax_arr).shape[-1]))
        spec = _validation_display_spec(raw, semantic, arr)
        key = _validation_virtual_key(side, semantic)
        n_comp = _infer_component_count(arr, vd, str(spec["location"]))
        vd.fields[key] = arr
        vd.field_info[key] = FieldInfo(
            key=key,
            label=f"{'JAX' if side == 'jax' else 'ABAQUS'} {spec['label']}",
            n_components=n_comp,
            location=str(spec["location"]),
            n_frames=_validation_frame_count(arr, vd, str(spec["location"])),
            symbol=str(spec.get("symbol", "")),
            formula=str(spec.get("formula", "")),
            description=str(spec.get("description", "")),
        )
    vd.metadata["validation_compare"] = True
    vd.metadata["validation_side"] = side


def _extract_metadata(raw: dict) -> dict:
    meta = raw.get("metadata")
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return {k: str(v) for k, v in meta.items()}
    if hasattr(meta, "dtype") and getattr(meta.dtype, "names", None):
        m = meta.flat[0] if meta.size else meta
        return {name: str(m[name]) for name in meta.dtype.names}
    return {}


def _coerce_text(value: Any) -> str:
    cur = _unwrap_scalar(value)
    if cur is None:
        return ""
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore").strip()
    return str(cur).strip()


def _extract_inp_hint(raw: dict) -> str:
    inp = raw.get("InpData")
    if inp is not None:
        try:
            if hasattr(inp, "dtype") and getattr(inp.dtype, "names", None):
                src = inp.flat[0] if inp.size == 1 else inp
                if "inp_path" in inp.dtype.names:
                    text = _coerce_text(src["inp_path"])
                    if text:
                        return text
            elif isinstance(inp, dict):
                text = _coerce_text(inp.get("inp_path"))
                if text:
                    return text
        except Exception:
            pass

    meta = raw.get("metadata")
    if meta is not None:
        try:
            if hasattr(meta, "dtype") and getattr(meta.dtype, "names", None):
                src = meta.flat[0] if meta.size == 1 else meta
                for key in ("source_inp", "source_file", "inp_path"):
                    if key in meta.dtype.names:
                        text = _coerce_text(src[key])
                        if text:
                            return text
            elif isinstance(meta, dict):
                for key in ("source_inp", "source_file", "inp_path"):
                    text = _coerce_text(meta.get(key))
                    if text:
                        return text
        except Exception:
            pass

    return ""


def _find_model_root(path: Path) -> Optional[Path]:
    resolved = path.resolve()
    for parent in resolved.parents:
        grand = parent.parent
        if grand and grand.name.lower() in {"examples", "storedmodels"}:
            return parent
    return None


def _guess_inp_candidates(mat_path: Path, raw: dict) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _push(path_like: Path) -> None:
        try:
            resolved = path_like.resolve()
        except Exception:
            return
        key = str(resolved).lower()
        if key in seen or not resolved.is_file():
            return
        seen.add(key)
        candidates.append(resolved)

    inp_hint = _extract_inp_hint(raw)
    if inp_hint:
        hint_path = Path(inp_hint)
        if not hint_path.is_absolute():
            hint_path = (mat_path.parent / hint_path)
        _push(hint_path)

    model_root = _find_model_root(mat_path)
    if model_root is not None:
        inp_files = sorted(model_root.rglob("*.inp"))
        if inp_files:
            load_match = re.search(r"(load\d+)", mat_path.stem, flags=re.IGNORECASE)
            if load_match:
                token = load_match.group(1).lower()
                matched = [path for path in inp_files if token in path.stem.lower()]
                if matched:
                    inp_files = matched
            for inp_path in inp_files:
                _push(inp_path)

    return candidates


@lru_cache(maxsize=32)
def _load_mesh_from_inp(inp_path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    from jaxmech.io.abaqus.inp import parse_inp
    from jaxmech.model.result import _mesh_to_viz_dict

    model = parse_inp(inp_path)
    mesh_dict = _mesh_to_viz_dict(model.mesh)
    points = mesh_dict.get("viz_points")
    cells = mesh_dict.get("viz_cells")
    cell_types = mesh_dict.get("viz_cell_types")
    ele_types = mesh_dict.get("viz_cell_block_ele_types")
    if points is None or cells is None or cell_types is None:
        return None, None, None, None
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(cells, dtype=np.int64),
        np.asarray(cell_types, dtype=np.int32),
        np.asarray(ele_types, dtype=object) if ele_types is not None else None,
    )


def _extract_mesh_from_model_fallback(mat_path: Path, raw: dict) -> tuple:
    """Fallback: recover mesh from a model-scoped INP when MAT lacks geometry."""
    for inp_path in _guess_inp_candidates(mat_path, raw):
        try:
            points, cells, cell_types, ele_types = _load_mesh_from_inp(str(inp_path))
        except Exception:
            continue
        if points is not None and cells is not None and cell_types is not None:
            return points, cells, cell_types, ele_types
    return None, None, None, None


def _infer_n_gauss_per_elem_from_raw(raw: dict) -> Optional[np.ndarray]:
    coords = _as_float_array(_lookup_any(raw, "gauss_coords"))
    if coords is not None and coords.ndim == 3 and coords.shape[-1] in (2, 3):
        return np.full(int(coords.shape[0]), int(coords.shape[1]), dtype=np.int32)
    return None


def _extract_mesh_from_gauss_points(raw: dict) -> tuple:
    """Build a point-cloud mesh from shakedown Gauss coordinates."""
    coords = _as_float_array(_lookup_any(raw, "gauss_coords"))
    if coords is None:
        return None, None, None, None
    if coords.ndim == 3 and coords.shape[-1] in (2, 3):
        points = coords.reshape(-1, coords.shape[-1])
    elif coords.ndim == 2 and coords.shape[-1] in (2, 3):
        points = coords
    else:
        return None, None, None, None
    n_points = int(points.shape[0])
    if n_points <= 0:
        return None, None, None, None
    cells = np.column_stack([
        np.ones(n_points, dtype=np.int64),
        np.arange(n_points, dtype=np.int64),
    ]).reshape(-1)
    cell_types = np.ones(n_points, dtype=np.int32)  # VTK_VERTEX
    ele_types = np.asarray(["GAUSS_POINT"] * n_points, dtype=object)
    return points.astype(np.float64), cells, cell_types, ele_types


def _alpha_scalar(value: Any) -> Optional[float]:
    arr = _as_float_array(value)
    if arr is None or arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _compute_shakedown_equilibrium_residual(parts: dict[str, Any]) -> Optional[np.ndarray]:
    eq_residual = _as_float_array(parts.get("equilibrium_residual"))
    if eq_residual is not None:
        return eq_residual.reshape(-1)
    residual = _as_float_array(parts.get("residual_stress"))
    c_sparse = parts.get("C_sparse")
    if residual is None or c_sparse is None:
        return None
    try:
        c_mat = c_sparse
        if isinstance(c_mat, np.ndarray) and c_mat.dtype == object and c_mat.size == 1:
            c_mat = c_mat.flat[0]
        if hasattr(c_mat, "toarray") or hasattr(c_mat, "dot"):
            vec = c_mat @ residual.reshape(-1)
        else:
            vec = np.asarray(c_mat, dtype=np.float64) @ residual.reshape(-1)
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


def _shakedown_equilibrium_node_field(
    raw: dict,
    vd: VizData,
    n_vert: int,
    parts: Optional[dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    if parts is None:
        parts = _shakedown_arrays(raw)
    eq_residual = _compute_shakedown_equilibrium_residual(parts)
    free_dofs = _free_dofs_array(parts)
    if eq_residual is None or free_dofs is None or eq_residual.size != free_dofs.size:
        return None

    if vd.points is not None:
        n_nodes = int(vd.points.shape[0])
        coord_dim = int(vd.points.shape[1])
    else:
        n_nodes, coord_dim = _shakedown_equilibrium_node_shape(parts)
    if n_nodes <= 0:
        return None

    n_global_dofs = int(np.max(free_dofs)) + 1
    ndof_per_node = max(coord_dim, int(np.ceil(n_global_dofs / n_nodes)))
    full = np.zeros(n_nodes * ndof_per_node, dtype=np.float64)
    valid = free_dofs < full.size
    if not np.any(valid):
        return None
    full[free_dofs[valid]] = eq_residual[valid]
    nodal = full.reshape(n_nodes, ndof_per_node)
    n_comp = min(coord_dim if coord_dim in (2, 3) else 3, nodal.shape[1])
    return nodal[:, :n_comp]


def _format_shakedown_weight_label(weights: np.ndarray) -> str:
    values = np.zeros(3, dtype=np.float64)
    flat = np.asarray(weights, dtype=np.float64).reshape(-1)
    take = min(3, flat.size)
    if take:
        values[:take] = flat[:take]

    def fmt(value: float) -> str:
        if abs(value) < 5.0e-13:
            value = 0.0
        if abs(value - round(value)) < 5.0e-10:
            return str(int(round(value)))
        return f"{value:.4g}"

    return "(" + ",".join(fmt(float(v)) for v in values) + ")"


def _format_shakedown_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if abs(value) < 5.0e-13:
        value = 0.0
    if abs(value - round(value)) < 5.0e-10:
        return str(int(round(value)))
    return f"{value:.4g}"


def _format_shakedown_vector(values: Any, n_values: Optional[int] = None) -> str:
    arr = _as_float_array(values)
    if arr is None or arr.size == 0:
        return "()"
    flat = arr.reshape(-1)
    if n_values is not None and n_values > 0:
        if flat.size < n_values:
            padded = np.zeros(int(n_values), dtype=np.float64)
            padded[: flat.size] = flat
            flat = padded
        flat = flat[:n_values]
    return "(" + ",".join(_format_shakedown_number(float(v)) for v in flat) + ")"


def _shakedown_load_dim(parts: dict[str, Any]) -> int:
    stress_cases = _shakedown_stress_cases(parts)
    if stress_cases is not None and stress_cases.ndim == 3:
        stress_dim = int(stress_cases.shape[2])
        if stress_dim > 1:
            return stress_dim
    layer_cases = _shakedown_layer_cases(parts)
    if layer_cases is not None and layer_cases.ndim == 4:
        layer_dim = int(layer_cases.shape[3])
        if layer_dim > 1:
            return layer_dim
    load_factor = _as_float_array(parts.get("load_factor"))
    if load_factor is not None and load_factor.size:
        flat = load_factor.reshape(-1)
        min_dim = 2 if flat.size >= 2 else 1
        while flat.size > min_dim and abs(float(flat[-1])) < 5.0e-13:
            flat = flat[:-1]
        return int(max(1, min(3, flat.size)))
    return 3


def _shakedown_display_load_vector(parts: dict[str, Any], index: int, n_dim: int) -> Optional[np.ndarray]:
    load_factor_set = _as_float_array(parts.get("load_factor_set"))
    use_explicit_load_set = load_factor_set is not None and load_factor_set.size > 0
    if not use_explicit_load_set:
        theta_values = _as_float_array(parts.get("theta_deg"))
        if theta_values is not None and theta_values.size > index:
            theta = float(theta_values.reshape(-1)[index])
            if np.isfinite(theta):
                if n_dim <= 1:
                    return np.asarray([1.0], dtype=np.float64)
                theta_rad = np.deg2rad(theta)
                if n_dim == 2:
                    return np.asarray([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64)
                phi_values = _as_float_array(parts.get("phi_deg"))
                if phi_values is not None and phi_values.size:
                    phi_flat = phi_values.reshape(-1)
                    phi = float(phi_flat[min(index, phi_flat.size - 1)])
                else:
                    phi = float(np.degrees(np.arcsin(1.0 / np.sqrt(3.0))))
                if np.isfinite(phi):
                    phi_rad = np.deg2rad(phi)
                    return np.asarray([
                        np.cos(phi_rad) * np.cos(theta_rad),
                        np.cos(phi_rad) * np.sin(theta_rad),
                        np.sin(phi_rad),
                    ], dtype=np.float64)

    load_factor = _as_float_array(parts.get("load_factor"))
    if load_factor is not None and load_factor.size:
        return load_factor.reshape(-1)
    return None


def _shakedown_vertex_labels(parts: dict[str, Any], n_vert: int) -> list[str]:
    order = _shakedown_vertex_display_order(parts, n_vert)
    return [
        str(_shakedown_vertex_detail(parts, original_idx, n_vert, display_index=display_idx).get("full") or "")
        for display_idx, original_idx in enumerate(order)
    ]


def _shakedown_display_vertex_load_matrix(parts: dict[str, Any]) -> Optional[np.ndarray]:
    """Rebuild user-facing vertex load coordinates for frame ordering.

    Some legacy MATs stored ``vertex_load_matrix`` in a solver-internal form
    that can disagree with the visible ``theta/load_factor`` definition.  The
    visualization label should describe the user-facing load point, while the
    stored matrix can still be used for numerical reconstruction.
    """
    stress_cases = _shakedown_stress_cases(parts)
    layer_cases = _shakedown_layer_cases(parts)
    if stress_cases is not None and stress_cases.ndim == 3:
        n_independent = int(stress_cases.shape[2])
    elif layer_cases is not None and layer_cases.ndim == 4:
        n_independent = int(layer_cases.shape[3])
    else:
        direct = _as_float_array(parts.get("vertex_load_matrix"))
        return direct if direct is not None and direct.ndim == 2 and direct.size else None
    if n_independent < 1 or n_independent > 3:
        return None
    n_independent = int(max(n_independent, min(3, _shakedown_load_dim(parts))))

    num_vert = _as_float_array(parts.get("NumVert"))
    n_vertices = int(num_vert.reshape(-1)[0]) if num_vert is not None and num_vert.size else None

    theta_values = _as_float_array(parts.get("theta_deg"))
    theta = _alpha_scalar(parts.get("theta_deg"))
    phi = _alpha_scalar(parts.get("phi_deg")) if n_independent == 3 else None
    has_theta = (
        theta is not None
        and np.isfinite(theta)
        and theta_values is not None
        and int(theta_values.size) == 1
    )
    if not has_theta:
        load_factor_set = _as_float_array(parts.get("load_factor_set"))
        if load_factor_set is not None and load_factor_set.ndim == 2 and load_factor_set.size:
            rows = load_factor_set[:, :n_independent]
            if n_vertices is None or rows.shape[0] == n_vertices:
                return np.asarray(rows.T, dtype=np.float64)

    load_scale = None
    if not has_theta:
        load_factor = _as_float_array(parts.get("load_factor"))
        if load_factor is not None and load_factor.size >= n_independent:
            load_scale = load_factor.reshape(-1)[:n_independent]
    r_ratios = _as_float_array(parts.get("R_ratios"))
    if r_ratios is None or not r_ratios.size:
        r_ratios = np.zeros(n_independent, dtype=np.float64)
    return _build_vertex_load_matrix_fallback(
        n_independent=n_independent,
        load_scale=load_scale,
        n_vertices=n_vertices,
        theta_deg=theta,
        phi_deg=phi,
        R_ratios=r_ratios.reshape(-1)[:n_independent],
    )


def _shakedown_vertex_display_order(parts: dict[str, Any], n_vertices: int) -> list[int]:
    n_vertices = int(max(1, n_vertices))
    order = list(range(n_vertices))
    vertex_load = _shakedown_display_vertex_load_matrix(parts)
    if vertex_load is None or vertex_load.ndim != 2 or vertex_load.shape[1] < 1:
        return order
    n_labels = min(int(vertex_load.shape[1]), n_vertices)
    coords = np.asarray(vertex_load[:, :n_labels], dtype=np.float64)
    spans = np.ptp(coords, axis=1) if coords.ndim == 2 and coords.size else np.asarray([])
    scale = max(1.0, float(np.nanmax(np.abs(coords))) if coords.size else 1.0)
    if spans.size and np.any(spans <= 1e-10 * scale):
        return [int(idx) for idx in range(n_labels)]
    order = sorted(
        range(n_labels),
        key=lambda idx: tuple(round(float(value), 12) for value in coords[:, idx]),
    )
    return [int(idx) for idx in order]


def _format_shakedown_result_label(
    parts: dict[str, Any],
    index: int,
    n_results: int,
    *,
    load_override: Optional[Any] = None,
) -> str:
    n_dim = _shakedown_load_dim(parts)
    load_factor = (
        _as_float_array(load_override)
        if load_override is not None
        else _shakedown_display_load_vector(parts, index, n_dim)
    )
    load_text = ""
    if load_factor is not None and np.asarray(load_factor).size:
        load_text = f"load = {_format_shakedown_vector(load_factor, n_dim)}"

    use_explicit_load_set = False
    load_factor_set = _as_float_array(parts.get("load_factor_set"))
    if load_factor_set is not None and load_factor_set.size:
        use_explicit_load_set = True

    prefix = f"Load point {index + 1}/{max(1, n_results)}"
    theta_values = _as_float_array(parts.get("theta_deg"))
    if (not use_explicit_load_set) and theta_values is not None and theta_values.size > index:
        theta = float(theta_values.reshape(-1)[index])
        if np.isfinite(theta):
            prefix = f"Theta {_format_shakedown_number(theta)} deg"
            phi_values = _as_float_array(parts.get("phi_deg"))
            if n_dim >= 3 and phi_values is not None and phi_values.size:
                phi_flat = phi_values.reshape(-1)
                phi = float(phi_flat[min(index, phi_flat.size - 1)])
                if np.isfinite(phi):
                    prefix += f", phi {_format_shakedown_number(phi)} deg"

    alpha = _alpha_scalar(parts.get("objective_value"))
    details = []
    if load_text:
        details.append(load_text)
    if alpha is not None and np.isfinite(alpha):
        details.append(f"alpha ={_format_shakedown_number(alpha)}")
    if details:
        return f"{prefix}; " + ", ".join(details)
    return prefix


def _shakedown_result_detail(
    parts: dict[str, Any],
    index: int,
    n_results: int,
    *,
    load_override: Optional[Any] = None,
) -> dict[str, Any]:
    n_dim = _shakedown_load_dim(parts)
    load_factor = (
        _as_float_array(load_override)
        if load_override is not None
        else _shakedown_display_load_vector(parts, index, n_dim)
    )
    load_factor_text = _format_shakedown_vector(load_factor, n_dim) if load_factor is not None else ""

    theta_text = ""
    theta_values = _as_float_array(parts.get("theta_deg"))
    load_factor_set = _as_float_array(parts.get("load_factor_set"))
    use_explicit_load_set = load_factor_set is not None and load_factor_set.size > 0
    if (not use_explicit_load_set) and theta_values is not None and theta_values.size > index:
        theta = float(theta_values.reshape(-1)[index])
        if np.isfinite(theta):
            theta_text = _format_shakedown_number(theta)

    phi_text = ""
    phi_values = _as_float_array(parts.get("phi_deg"))
    if n_dim >= 3 and phi_values is not None and phi_values.size:
        phi_flat = phi_values.reshape(-1)
        phi = float(phi_flat[min(index, phi_flat.size - 1)])
        if np.isfinite(phi):
            phi_text = _format_shakedown_number(phi)

    alpha = _alpha_scalar(parts.get("objective_value"))
    alpha_text = _format_shakedown_number(alpha) if alpha is not None and np.isfinite(alpha) else ""
    return {
        "result_index": int(index),
        "result_count": int(max(1, n_results)),
        "load_factor": load_factor_text,
        "theta_deg": theta_text,
        "phi_deg": phi_text,
        "alpha": alpha_text,
        "full": _format_shakedown_result_label(parts, index, n_results, load_override=load_override),
    }


def _shakedown_vertex_detail(
    parts: dict[str, Any],
    index: int,
    n_vertices: int,
    *,
    display_index: Optional[int] = None,
) -> dict[str, Any]:
    vertex_load = _shakedown_display_vertex_load_matrix(parts)
    if vertex_load is None:
        vertex_load = _shakedown_vertex_load_matrix(parts)
    load_values: list[float] = []
    load_text = ""
    if vertex_load is not None and vertex_load.ndim == 2 and vertex_load.shape[1] > index:
        n_dim = _shakedown_load_dim(parts)
        values = np.asarray(vertex_load[:, index], dtype=np.float64).reshape(-1)
        load_values = [float(value) for value in values]
        load_text = _format_shakedown_vector(values, n_dim)
    shown_index = int(index if display_index is None else display_index)
    full = f"Vertex {shown_index + 1}/{max(1, n_vertices)}"
    return {
        "vertex_index": shown_index,
        "vertex_source_index": int(index),
        "vertex_count": int(max(1, n_vertices)),
        "vertex_load": load_text,
        "vertex_load_values": load_values,
        "full": full,
    }


def _shakedown_vertex_result_label_and_detail(
    parts: dict[str, Any],
    result_index: int,
    n_results: int,
    vertex_detail: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    load_override = vertex_detail.get("vertex_load_values") or None
    return (
        _format_shakedown_result_label(
            parts,
            result_index,
            n_results,
            load_override=load_override,
        ),
        _shakedown_result_detail(
            parts,
            result_index,
            n_results,
            load_override=load_override,
        ),
    )


def _shakedown_frame_detail(
    result_detail: dict[str, Any],
    *,
    vertex_detail: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    load_factor = str(result_detail.get("load_factor") or "")
    vertex_text = ""
    if vertex_detail:
        vertex_text = f"{int(vertex_detail.get('vertex_index', 0)) + 1}/{int(vertex_detail.get('vertex_count', 1))}"
    full_parts = [str(result_detail.get("full") or "")]
    if vertex_detail and vertex_detail.get("full"):
        full_parts.append(str(vertex_detail["full"]))
    detail = {
        "load_factor": load_factor,
        "vertex": vertex_text,
        "full": " - ".join(part for part in full_parts if part),
        "result_index": int(result_detail.get("result_index", 0)),
        "result_count": int(result_detail.get("result_count", 1)),
        "alpha": str(result_detail.get("alpha") or ""),
        "theta_deg": str(result_detail.get("theta_deg") or ""),
        "phi_deg": str(result_detail.get("phi_deg") or ""),
    }
    return detail


_RSDMS_VERTEX_FRAME_KEYS = {
    "rsdms_S_tot",
    "rsdms_CInEQ",
    "rsdms_S_tot_SNEG",
    "rsdms_S_tot_SPOS",
    "rsdms_E_tot_SNEG",
    "rsdms_E_tot_SPOS",
    "rsdms_CInEQ_SNEG",
    "rsdms_CInEQ_SPOS",
    "rsdms_SF_tot",
    "rsdms_SM_tot",
    "rsdms_GE_tot",
    "rsdms_GK_tot",
    "rsdms_NFORC",
    "rsdms_FERROR",
}

_RSDMS_PATH_FRAME_KEYS = {
    "rsdms_XS_path",
    "rsdms_XS_path_SNEG",
    "rsdms_XS_path_SPOS",
}


def _is_rsdms_field_key(key: str) -> bool:
    return str(key).startswith("rsdms_")


def _selection_has_rsdms_fields(selected: set[str]) -> bool:
    return any(_is_rsdms_field_key(key) for key in selected)


def _selection_has_rsdm_fields(selected: set[str]) -> bool:
    return any(str(key).startswith("rsdm_") for key in selected)


def _rsdms_vertex_path_order(vertex_load_matrix: np.ndarray) -> list[int]:
    vertices = np.asarray(vertex_load_matrix, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] == 0:
        return []
    n_cases, n_vert = int(vertices.shape[0]), int(vertices.shape[1])
    if n_cases == 2 and n_vert == 4:
        lo = np.min(vertices, axis=1)
        hi = np.max(vertices, axis=1)
        targets = [
            np.array([lo[0], lo[1]], dtype=np.float64),
            np.array([hi[0], lo[1]], dtype=np.float64),
            np.array([hi[0], hi[1]], dtype=np.float64),
            np.array([lo[0], hi[1]], dtype=np.float64),
        ]
        order: list[int] = []
        used: set[int] = set()
        for target in targets:
            dist = np.linalg.norm(vertices.T - target.reshape(1, -1), axis=1)
            for idx in np.argsort(dist):
                idx = int(idx)
                if idx not in used:
                    order.append(idx)
                    used.add(idx)
                    break
        if len(order) == 4:
            return order + [order[0]]
    if n_vert == 2:
        return [0, 1, 0]
    return list(range(n_vert)) + [0]


def _shakedown_yield_stress(parts: Mapping[str, Any]) -> Optional[float]:
    for key in ("yield_stress", "yield_values"):
        value = parts.get(key)
        arr = _as_float_array(value)
        if arr is not None and arr.size:
            candidate = float(arr.reshape(-1)[0])
            if np.isfinite(candidate):
                return candidate
        text = _coerce_text(value)
        if text:
            numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            for item in numbers:
                try:
                    candidate = float(item)
                except ValueError:
                    continue
                if np.isfinite(candidate):
                    return candidate
    return None


def _rsdms_xs_path_frames_and_metadata(
    parts: Mapping[str, Any],
    *,
    result_index: int,
    result_count: int,
) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    ratio_raw = _as_float_array(parts.get("yield_ratio"), squeeze=False)
    vertex_load = _shakedown_display_vertex_load_matrix(dict(parts))
    if vertex_load is None:
        vertex_load = _shakedown_vertex_load_matrix(dict(parts))
    yield_stress = _shakedown_yield_stress(parts)
    if ratio_raw is None or vertex_load is None or yield_stress is None:
        return [], [], []

    ratio = np.asarray(np.squeeze(ratio_raw), dtype=np.float64)
    if ratio.ndim != 2 or ratio.size == 0:
        return [], [], []
    n_vert = int(vertex_load.shape[1])
    if ratio.shape[1] != n_vert and ratio.shape[0] == n_vert:
        ratio = ratio.T
    if ratio.shape[1] != n_vert:
        return [], [], []

    order = _rsdms_vertex_path_order(vertex_load)
    if len(order) < 2:
        return [], [], []

    row_label = _format_shakedown_result_label(dict(parts), result_index, result_count)
    row_detail = _shakedown_result_detail(dict(parts), result_index, result_count)
    n_dim = _shakedown_load_dim(dict(parts))
    frames: list[np.ndarray] = []
    labels: list[str] = []
    details: list[dict[str, Any]] = []
    n_paths = len(order) - 1
    for path_idx, (start_idx, end_idx) in enumerate(zip(order[:-1], order[1:]), start=1):
        if start_idx >= ratio.shape[1] or end_idx >= ratio.shape[1]:
            continue
        segment = np.maximum(np.maximum(ratio[:, start_idx], ratio[:, end_idx]) - 1.0, 0.0)
        frames.append(segment * float(yield_stress))
        start_text = _format_shakedown_vector(vertex_load[:, start_idx], n_dim)
        end_text = _format_shakedown_vector(vertex_load[:, end_idx], n_dim)
        path_label = f"Path {path_idx}/{n_paths}: load {start_text} -> {end_text}"
        labels.append(f"{row_label} - {path_label}")
        detail = _shakedown_frame_detail(row_detail)
        detail["path"] = f"{path_idx}/{n_paths}"
        detail["path_load"] = f"{start_text} -> {end_text}"
        detail["full"] = f"{detail.get('full') or row_label} - {path_label}"
        details.append(detail)
    return frames, labels, details


def _rsdms_xs_path_surface_frames_and_metadata(
    parts: Mapping[str, Any],
    *,
    surface: str,
    result_index: int,
    result_count: int,
) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    ratio_surface = _shakedown_layer_yield_surface(dict(parts), surface)
    vertex_load = _shakedown_display_vertex_load_matrix(dict(parts))
    if vertex_load is None:
        vertex_load = _shakedown_vertex_load_matrix(dict(parts))
    yield_stress = _shakedown_yield_stress(parts)
    if ratio_surface is None or vertex_load is None or yield_stress is None:
        return [], [], []
    ratio = np.asarray(ratio_surface, dtype=np.float64).T
    if ratio.ndim != 2 or ratio.shape[1] != int(vertex_load.shape[1]):
        return [], [], []
    order = _rsdms_vertex_path_order(vertex_load)
    if len(order) < 2:
        return [], [], []

    row_label = _format_shakedown_result_label(dict(parts), result_index, result_count)
    row_detail = _shakedown_result_detail(dict(parts), result_index, result_count)
    n_dim = _shakedown_load_dim(dict(parts))
    frames: list[np.ndarray] = []
    labels: list[str] = []
    details: list[dict[str, Any]] = []
    n_paths = len(order) - 1
    surface_label = str(surface).upper()
    for path_idx, (start_idx, end_idx) in enumerate(zip(order[:-1], order[1:]), start=1):
        segment = np.maximum(np.maximum(ratio[:, start_idx], ratio[:, end_idx]) - 1.0, 0.0)
        frames.append(segment * float(yield_stress))
        start_text = _format_shakedown_vector(vertex_load[:, start_idx], n_dim)
        end_text = _format_shakedown_vector(vertex_load[:, end_idx], n_dim)
        path_label = f"Path {path_idx}/{n_paths}: load {start_text} -> {end_text}"
        labels.append(f"{row_label} - {surface_label} - {path_label}")
        detail = _shakedown_frame_detail(row_detail)
        detail["path"] = f"{path_idx}/{n_paths}"
        detail["path_load"] = f"{start_text} -> {end_text}"
        detail["surface"] = surface_label
        detail["full"] = f"{detail.get('full') or row_label} - {surface_label} - {path_label}"
        details.append(detail)
    return frames, labels, details


def _register_rsdms_path_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    result_parts = _shakedown_result_parts(raw)
    result_parts = [parts for parts in result_parts if _shakedown_counts(parts)[2] > 0]
    if not result_parts:
        return

    n_results = len(result_parts)
    specs = [
        ("rsdms_XS_path", "", "RSDM-S path effective excess stress"),
        ("rsdms_XS_path_SNEG", "SNEG", "RSDM-S XS_path (inner/SNEG)"),
        ("rsdms_XS_path_SPOS", "SPOS", "RSDM-S XS_path (outer/SPOS)"),
    ]
    for key, surface, label in specs:
        if selected_fields is not None and key not in selected_fields:
            continue
        if key in vd.field_info:
            continue
        frames: list[np.ndarray] = []
        labels: list[str] = []
        details: list[dict[str, Any]] = []
        for idx, parts in enumerate(result_parts):
            if surface:
                row_frames, row_labels, row_details = _rsdms_xs_path_surface_frames_and_metadata(
                    parts,
                    surface=surface,
                    result_index=idx,
                    result_count=n_results,
                )
            else:
                row_frames, row_labels, row_details = _rsdms_xs_path_frames_and_metadata(
                    parts,
                    result_index=idx,
                    result_count=n_results,
                )
            frames.extend(row_frames)
            labels.extend(row_labels)
            details.extend(row_details)
        if not frames:
            continue

        data = np.stack(frames, axis=0)
        display_meta = _FIELD_DISPLAY_METADATA[key]
        vd.fields[key] = data
        vd.field_info[key] = FieldInfo(
            key=key,
            label=label,
            n_components=1,
            location="gauss",
            n_frames=int(data.shape[0]),
            symbol=display_meta.get("symbol", ""),
            formula=display_meta.get("formula", ""),
            description=display_meta.get("description", ""),
        )
        by_field = vd.metadata.get("shakedown_frame_labels_by_field", {})
        if not isinstance(by_field, dict):
            by_field = {}
        by_field[key] = labels
        vd.metadata["shakedown_frame_labels_by_field"] = by_field

        details_by_field = vd.metadata.get("shakedown_frame_details_by_field", {})
        if not isinstance(details_by_field, dict):
            details_by_field = {}
        details_by_field[key] = details
        vd.metadata["shakedown_frame_details_by_field"] = details_by_field


def _stress_vm(stress: np.ndarray) -> np.ndarray:
    arr = np.asarray(stress, dtype=np.float64)
    if arr.ndim >= 1 and arr.shape[-1] >= 6:
        s11, s22, s33, s12, s23, s13 = [arr[..., i] for i in range(6)]
        return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (s12 ** 2 + s23 ** 2 + s13 ** 2))
    if arr.ndim >= 1 and arr.shape[-1] >= 3:
        s11, s22, s12 = arr[..., 0], arr[..., 1], arr[..., 2]
        return np.sqrt(np.maximum(s11 ** 2 - s11 * s22 + s22 ** 2 + 3.0 * s12 ** 2, 0.0))
    return np.asarray(arr, dtype=np.float64)


def _rsdm_path_count(raw: Mapping[str, Any], n_time: int) -> int:
    for value in (
        _struct_get(raw.get("RSDMInput"), "n_vert"),
        _struct_get(raw.get("RSDMShakedownInput"), "n_vert"),
        _lookup_any(dict(raw), "NumVert"),
    ):
        arr = _as_float_array(value)
        if arr is not None and arr.size:
            candidate = int(arr.reshape(-1)[0])
            if candidate > 0:
                return max(1, min(candidate, int(n_time)))
    return max(1, min(1, int(n_time)))


def _rsdm_cycle_array(raw: Mapping[str, Any], name: str) -> Optional[np.ndarray]:
    value = _lookup_any(dict(raw), name)
    arr = _as_float_array(value, squeeze=False)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim >= 3:
        return arr.reshape(arr.shape[0], -1, arr.shape[-1])
    if arr.ndim == 2 and arr.shape[-1] in (3, 4, 6):
        return arr.reshape(1, arr.shape[0], arr.shape[1])
    return None


def _rsdm_time_grid(raw: Mapping[str, Any], n_time: int) -> np.ndarray:
    value = _lookup_any(dict(raw), "time_grid")
    arr = _as_float_array(value)
    if arr is not None and arr.size:
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if flat.size == n_time:
            return flat
    return np.linspace(0.0, 1.0, max(1, int(n_time)), dtype=np.float64)


def _rsdm_integral_from_cycle(cycle: np.ndarray, raw: Mapping[str, Any]) -> np.ndarray:
    n_time = int(cycle.shape[0])
    if n_time <= 1:
        return np.zeros_like(cycle[0])
    time = _rsdm_time_grid(raw, n_time)
    try:
        return np.trapz(cycle, x=time, axis=0)
    except Exception:
        return np.mean(cycle, axis=0)


def _register_field(
    vd: VizData,
    key: str,
    data: np.ndarray,
    *,
    label: str,
    location: str,
    n_components: int,
    n_frames: int = 1,
) -> None:
    display_meta = _FIELD_DISPLAY_METADATA.get(key, {})
    vd.fields[key] = data
    vd.field_info[key] = FieldInfo(
        key=key,
        label=label,
        n_components=int(n_components),
        location=location,
        n_frames=int(n_frames),
        symbol=display_meta.get("symbol", ""),
        formula=display_meta.get("formula", ""),
        description=display_meta.get("description", ""),
    )


def _register_rsdm_steady_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    requested = {
        "rsdm_S_res_a0",
        "rsdm_S_res_cyc",
        "rsdm_S_tot_cyc",
        "rsdm_XS_cyc",
        "rsdm_XS_int",
        "rsdm_XS_norm",
        "rsdm_XS_cmax",
        "rsdm_XS_max",
        "rsdm_XS_path",
        "rsdm_XS_vmmax",
        "rsdm_CLS_ratio",
        "rsdm_CLS_map",
    }
    if selected_fields is not None:
        requested.intersection_update(selected_fields)
    if not requested:
        return

    residual_a0 = _as_float_array(_lookup_any(raw, "residual_stress_a0"), squeeze=False)
    residual_cycle = _rsdm_cycle_array(raw, "residual_stress_cycle")
    total_cycle = _rsdm_cycle_array(raw, "total_stress_cycle")
    excess_cycle = _rsdm_cycle_array(raw, "excess_stress_cycle")

    n_comp = 1
    for arr in (residual_cycle, total_cycle, excess_cycle, residual_a0):
        if arr is not None and arr.ndim >= 1:
            n_comp = int(arr.shape[-1])
            break

    if "rsdm_S_res_a0" in requested and residual_a0 is not None:
        arr = np.asarray(residual_a0, dtype=np.float64)
        if arr.ndim >= 3:
            arr = arr.reshape(-1, arr.shape[-1])
        _register_field(vd, "rsdm_S_res_a0", arr, label="Residual stress mean", location="gauss", n_components=n_comp)
    if "rsdm_S_res_cyc" in requested and residual_cycle is not None:
        _register_field(vd, "rsdm_S_res_cyc", residual_cycle, label="Residual stress history", location="gauss", n_components=n_comp, n_frames=residual_cycle.shape[0])
    if "rsdm_S_tot_cyc" in requested and total_cycle is not None:
        _register_field(vd, "rsdm_S_tot_cyc", total_cycle, label="Total stress history", location="gauss", n_components=n_comp, n_frames=total_cycle.shape[0])
    if "rsdm_XS_cyc" in requested and excess_cycle is not None:
        _register_field(vd, "rsdm_XS_cyc", excess_cycle, label="Excess stress history", location="gauss", n_components=n_comp, n_frames=excess_cycle.shape[0])

    if excess_cycle is None:
        return

    vm = _stress_vm(excess_cycle)
    integral = _rsdm_integral_from_cycle(excess_cycle, raw)
    alpha_norm = np.linalg.norm(integral, axis=-1)
    cmax = np.max(np.abs(excess_cycle), axis=0)
    xmax = np.max(np.abs(excess_cycle), axis=(0, 2))
    vmmax = np.max(vm, axis=0)
    denom = np.maximum(vmmax, 1.0e-30)
    ratio = alpha_norm / denom
    tol = _as_float_array(_lookup_any(raw, "classification_rel_tol"))
    rel_tol = float(tol.reshape(-1)[0]) if tol is not None and tol.size else 1.0e-2
    state_map = np.where(vmmax <= 1.0e-14, 0, np.where(ratio > rel_tol, 2, 1)).astype(np.float64)

    derived_specs = [
        ("rsdm_XS_int", integral, "Alpha integral", n_comp),
        ("rsdm_XS_norm", alpha_norm.reshape(-1, 1), "Alpha norm", 1),
        ("rsdm_XS_cmax", cmax, "Max |sigma_p component|", n_comp),
        ("rsdm_XS_max", xmax.reshape(-1, 1), "Max |sigma_p|", 1),
        ("rsdm_XS_vmmax", vmmax.reshape(-1, 1), "Max effective excess stress", 1),
        ("rsdm_CLS_ratio", ratio.reshape(-1, 1), "Classification ratio", 1),
        ("rsdm_CLS_map", state_map.reshape(-1, 1), "State classification", 1),
    ]
    for key, data, label, comps in derived_specs:
        if key in requested:
            _register_field(vd, key, data, label=label, location="gauss", n_components=comps)

    if "rsdm_XS_path" in requested:
        n_paths = _rsdm_path_count(raw, int(vm.shape[0]))
        frames = [np.max(chunk, axis=0) for chunk in np.array_split(vm, n_paths) if chunk.size]
        if frames:
            data = np.stack(frames, axis=0)
            _register_field(vd, "rsdm_XS_path", data, label="XS_path", location="gauss", n_components=1, n_frames=data.shape[0])
            labels = [f"Path {idx}/{len(frames)}" for idx in range(1, len(frames) + 1)]
            by_field = vd.metadata.get("shakedown_frame_labels_by_field", {})
            if not isinstance(by_field, dict):
                by_field = {}
            by_field["rsdm_XS_path"] = labels
            vd.metadata["shakedown_frame_labels_by_field"] = by_field


def _rsdms_yield_stress_for_parts(parts: Mapping[str, Any]) -> float:
    value = _shakedown_yield_stress(parts)
    return float(value) if value is not None and np.isfinite(value) else 1.0


def _rsdms_generalized_strain_component(parts: dict[str, Any], *, total: bool, suffix: str) -> Optional[np.ndarray]:
    residual = _normalize_generalized_cases(parts.get("residual_generalized_strain"))
    offset = 0 if str(suffix).upper() == "GE" else 3
    if residual is None or residual.shape[1] < offset + 3:
        return None
    residual_comp = residual[:, offset:offset + 3, 0]
    if not total:
        return residual_comp
    cases = _normalize_generalized_cases(parts.get("shell_generalized_strain_cases"))
    vertex_load = _shakedown_vertex_load_matrix(parts)
    alpha = _alpha_scalar(parts.get("objective_value"))
    if cases is None or vertex_load is None or alpha is None:
        return None
    if cases.shape[0] != residual.shape[0] or cases.shape[1] < offset + 3 or cases.shape[2] != vertex_load.shape[0]:
        return None
    elastic_vertices = np.einsum("gci,iv->gcv", cases[:, offset:offset + 3, :], vertex_load)
    return residual_comp[np.newaxis, :, :] + float(alpha) * np.moveaxis(elastic_vertices, 2, 0)


def _register_rsdms_summary_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    requested = {
        "rsdms_S_res", "rsdms_S_tot", "rsdms_CInEQ", "rsdms_CEQ", "rsdms_XS_vmmax",
        "rsdms_S_res_SNEG", "rsdms_S_res_SPOS", "rsdms_S_tot_SNEG", "rsdms_S_tot_SPOS",
        "rsdms_E_res_SNEG", "rsdms_E_res_SPOS", "rsdms_E_tot_SNEG", "rsdms_E_tot_SPOS",
        "rsdms_CInEQ_SNEG", "rsdms_CInEQ_SPOS",
        "rsdms_SF_res", "rsdms_SM_res", "rsdms_SF_tot", "rsdms_SM_tot",
        "rsdms_GE_res", "rsdms_GK_res", "rsdms_GE_tot", "rsdms_GK_tot",
        "rsdms_NFORC", "rsdms_FERROR",
    }
    if selected_fields is not None:
        requested.intersection_update(selected_fields)
    if not requested:
        return
    result_parts = _shakedown_result_parts(raw)
    result_parts = [parts for parts in result_parts if _shakedown_counts(parts)[2] > 0]
    if not result_parts:
        return

    frames: dict[str, list[np.ndarray]] = {key: [] for key in requested}
    for parts in result_parts:
        if not _shakedown_is_shell_layer(parts):
            residual = _as_float_array(parts.get("residual_stress"))
            stress_cases = _shakedown_stress_cases(parts)
            vertex_load = _shakedown_vertex_load_matrix(parts)
            alpha = _alpha_scalar(parts.get("objective_value"))
            if "rsdms_S_res" in requested and residual is not None and residual.ndim == 2:
                frames["rsdms_S_res"].append(residual)
            if (
                "rsdms_S_tot" in requested and residual is not None and stress_cases is not None
                and vertex_load is not None and alpha is not None and stress_cases.shape[2] == vertex_load.shape[0]
            ):
                elastic_vertices = np.einsum("gsi,iv->gsv", stress_cases, vertex_load)
                total = residual[np.newaxis, :, :] + float(alpha) * np.moveaxis(elastic_vertices, 2, 0)
                frames["rsdms_S_tot"].extend(total[i] for i in range(total.shape[0]))
            ratio_frames = _shakedown_yield_ratio_frames(parts)
            if ratio_frames is not None:
                violation = np.maximum(ratio_frames - 1.0, 0.0)
                if "rsdms_CInEQ" in requested:
                    frames["rsdms_CInEQ"].extend(violation[i] for i in range(violation.shape[0]))
                if "rsdms_XS_vmmax" in requested:
                    frames["rsdms_XS_vmmax"].append(np.max(violation, axis=0) * _rsdms_yield_stress_for_parts(parts))
            eq = _as_float_array(parts.get("equilibrium_residual"))
            if "rsdms_CEQ" in requested and eq is not None:
                n_nodes, n_dof = _shakedown_equilibrium_node_shape(parts)
                if n_nodes > 0:
                    frames["rsdms_CEQ"].append(
                        _scatter_free_vector_to_nodes(eq.reshape(-1), parts.get("free_dofs"), n_nodes=n_nodes, ndof_per_node=n_dof)
                    )
            continue

        for surface in ("SNEG", "SPOS"):
            residual_surface = _shakedown_layer_residual_surface(parts, surface)
            total_surface = _shakedown_layer_total_surface(parts, surface)
            yield_surface = _shakedown_layer_yield_surface(parts, surface)
            for key, value in (
                (f"rsdms_S_res_{surface}", residual_surface),
                (f"rsdms_S_tot_{surface}", total_surface),
            ):
                if key in requested and value is not None:
                    frames[key].extend(value[i] for i in range(value.shape[0])) if value.ndim == 3 else frames[key].append(value)
            if f"rsdms_CInEQ_{surface}" in requested and yield_surface is not None:
                violation = np.maximum(yield_surface - 1.0, 0.0)
                frames[f"rsdms_CInEQ_{surface}"].extend(violation[i] for i in range(violation.shape[0]))
        for suffix in ("SF", "SM"):
            residual_gen = _shakedown_generalized_component(parts, total=False, suffix=suffix)
            total_gen = _shakedown_generalized_component(parts, total=True, suffix=suffix)
            if f"rsdms_{suffix}_res" in requested and residual_gen is not None:
                frames[f"rsdms_{suffix}_res"].append(residual_gen)
            if f"rsdms_{suffix}_tot" in requested and total_gen is not None:
                frames[f"rsdms_{suffix}_tot"].extend(total_gen[i] for i in range(total_gen.shape[0]))
        for suffix in ("GE", "GK"):
            residual_strain = _rsdms_generalized_strain_component(parts, total=False, suffix=suffix)
            total_strain = _rsdms_generalized_strain_component(parts, total=True, suffix=suffix)
            if f"rsdms_{suffix}_res" in requested and residual_strain is not None:
                frames[f"rsdms_{suffix}_res"].append(residual_strain)
            if f"rsdms_{suffix}_tot" in requested and total_strain is not None:
                frames[f"rsdms_{suffix}_tot"].extend(total_strain[i] for i in range(total_strain.shape[0]))
        for key, part_key in (("rsdms_NFORC", "shell_nforc"), ("rsdms_FERROR", "shell_force_error")):
            arr = _as_float_array(parts.get(part_key), squeeze=False)
            if key in requested and arr is not None:
                arr = np.squeeze(np.asarray(arr, dtype=np.float64))
                if arr.ndim >= 3:
                    frames[key].extend(arr[i] for i in range(arr.shape[0]))
                elif arr.ndim == 2:
                    frames[key].append(arr)

    labels: dict[str, tuple[str, str, int]] = {
        "rsdms_S_res": ("RSDM-S residual stress", "gauss", 6),
        "rsdms_S_tot": ("RSDM-S total stress", "gauss", 6),
        "rsdms_CInEQ": ("RSDM-S CInEQ", "gauss", 1),
        "rsdms_CEQ": ("RSDM-S CEQ", "node", 3),
        "rsdms_XS_vmmax": ("RSDM-S max effective excess stress", "gauss", 1),
        "rsdms_NFORC": ("RSDM-S NFORC (f_int+f_drill)", "node", 6),
        "rsdms_FERROR": ("RSDM-S FERROR", "node", 6),
    }
    for surface, label_suffix in (("SNEG", "inner/SNEG"), ("SPOS", "outer/SPOS")):
        labels.update({
            f"rsdms_S_res_{surface}": (f"RSDM-S S residual ({label_suffix})", "gauss", 3),
            f"rsdms_S_tot_{surface}": (f"RSDM-S S total ({label_suffix})", "gauss", 3),
            f"rsdms_E_res_{surface}": (f"RSDM-S E residual-equivalent ({label_suffix})", "gauss", 3),
            f"rsdms_E_tot_{surface}": (f"RSDM-S E total ({label_suffix})", "gauss", 3),
            f"rsdms_CInEQ_{surface}": (f"RSDM-S CInEQ ({label_suffix})", "gauss", 1),
        })
    labels.update({
        "rsdms_SF_res": ("RSDM-S SF residual", "gauss", 3),
        "rsdms_SM_res": ("RSDM-S SM residual", "gauss", 3),
        "rsdms_SF_tot": ("RSDM-S SF total", "gauss", 3),
        "rsdms_SM_tot": ("RSDM-S SM total", "gauss", 3),
        "rsdms_GE_res": ("RSDM-S GE residual-equivalent", "gauss", 3),
        "rsdms_GK_res": ("RSDM-S GK residual-equivalent", "gauss", 3),
        "rsdms_GE_tot": ("RSDM-S GE total", "gauss", 3),
        "rsdms_GK_tot": ("RSDM-S GK total", "gauss", 3),
    })
    for key, items in frames.items():
        if not items:
            continue
        data = np.stack(items, axis=0) if len(items) > 1 else items[0]
        n_frames = int(data.shape[0]) if data.ndim >= 3 else 1
        label, loc, fallback_comps = labels.get(key, (key, "gauss", 1))
        comps = int(data.shape[-1]) if data.ndim >= 2 and data.shape[-1] in (1, 2, 3, 4, 6) else fallback_comps
        _register_field(vd, key, data, label=label, location=loc, n_components=comps, n_frames=n_frames)


def _register_rsdm_path_field(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    key = "rsdm_XS_path"
    if selected_fields is not None and key not in selected_fields:
        return
    if key in vd.field_info:
        return
    cycle = None
    for source_key in ("rsdm_XS_cyc", "excess_stress_cycle", "field_excess_stress_cycle"):
        if raw.get(source_key) is not None:
            cycle = raw.get(source_key)
            break
    if cycle is None:
        return
    arr = np.asarray(cycle, dtype=np.float64)
    if arr.size == 0:
        return
    if arr.ndim >= 3:
        flat = arr.reshape(arr.shape[0], -1, arr.shape[-1])
        vm = _stress_vm(flat)
    elif arr.ndim == 2:
        vm = arr
    else:
        return
    n_time = int(vm.shape[0])
    n_paths = _rsdm_path_count(raw, n_time)
    frames = [np.max(chunk, axis=0) for chunk in np.array_split(vm, n_paths) if chunk.size]
    if not frames:
        return
    data = np.stack(frames, axis=0)
    display_meta = _FIELD_DISPLAY_METADATA[key]
    vd.fields[key] = data
    vd.field_info[key] = FieldInfo(
        key=key,
        label="XS_path",
        n_components=1,
        location="gauss",
        n_frames=int(data.shape[0]),
        symbol=display_meta.get("symbol", ""),
        formula=display_meta.get("formula", ""),
        description=display_meta.get("description", ""),
    )
    labels = [f"Path {idx}/{len(frames)}" for idx in range(1, len(frames) + 1)]
    by_field = vd.metadata.get("shakedown_frame_labels_by_field", {})
    if not isinstance(by_field, dict):
        by_field = {}
    by_field[key] = labels
    vd.metadata["shakedown_frame_labels_by_field"] = by_field


def _enrich_rsdms_summary_frame_metadata(raw: dict, vd: VizData) -> None:
    """Attach load-point and vertex labels for direct-method RSDM-S fields."""
    rsdms_keys = [key for key in vd.field_info if _is_rsdms_field_key(key)]
    if not rsdms_keys:
        return

    result_parts = _shakedown_result_parts(raw)
    result_parts = [parts for parts in result_parts if _shakedown_counts(parts)[2] > 0]
    if not result_parts:
        return

    n_results = len(result_parts)
    labels_by_field: dict[str, list[str]] = {}
    details_by_field: dict[str, list[dict[str, Any]]] = {}

    for key in rsdms_keys:
        fi = vd.field_info.get(key)
        if fi is None:
            continue

        labels: list[str] = []
        details: list[dict[str, Any]] = []
        if key in _RSDMS_PATH_FRAME_KEYS:
            surface = ""
            if key.endswith("_SNEG"):
                surface = "SNEG"
            elif key.endswith("_SPOS"):
                surface = "SPOS"
            for idx, parts in enumerate(result_parts):
                if surface:
                    _frames, row_labels, row_details = _rsdms_xs_path_surface_frames_and_metadata(
                        parts,
                        surface=surface,
                        result_index=idx,
                        result_count=n_results,
                    )
                else:
                    _frames, row_labels, row_details = _rsdms_xs_path_frames_and_metadata(
                        parts,
                        result_index=idx,
                        result_count=n_results,
                    )
                labels.extend(row_labels)
                details.extend(row_details)
        elif key in _RSDMS_VERTEX_FRAME_KEYS:
            field_data = vd.fields.get(key)
            frame_order: list[int] = []
            for idx, parts in enumerate(result_parts):
                _, _, n_vert = _shakedown_counts(parts)
                row_label = _format_shakedown_result_label(parts, idx, n_results)
                row_detail = _shakedown_result_detail(parts, idx, n_results)
                display_order = _shakedown_vertex_display_order(parts, n_vert)
                for display_idx, frame_idx in enumerate(display_order):
                    vertex_detail = _shakedown_vertex_detail(
                        parts, frame_idx, n_vert, display_index=display_idx
                    )
                    vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                        parts, idx, n_results, vertex_detail
                    )
                    source_frame = idx * n_vert + frame_idx
                    frame_order.append(source_frame)
                    vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{n_vert}")
                    labels.append(f"{vertex_row_label} - {vertex_label}")
                    details.append(_shakedown_frame_detail(
                        vertex_row_detail,
                        vertex_detail=vertex_detail,
                    ))
            if (
                isinstance(field_data, np.ndarray)
                and field_data.ndim >= 1
                and len(frame_order) == int(fi.n_frames)
                and max(frame_order, default=-1) < int(field_data.shape[0])
            ):
                vd.fields[key] = field_data[np.asarray(frame_order, dtype=np.int64), ...]
        else:
            for idx, parts in enumerate(result_parts):
                row_label = _format_shakedown_result_label(parts, idx, n_results)
                row_detail = _shakedown_result_detail(parts, idx, n_results)
                labels.append(row_label)
                details.append(_shakedown_frame_detail(row_detail))

        n_frames = int(fi.n_frames)
        if len(labels) >= n_frames:
            labels_by_field[key] = labels[:n_frames]
            details_by_field[key] = details[:n_frames]

    if labels_by_field:
        existing = vd.metadata.get("shakedown_frame_labels_by_field", {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update(labels_by_field)
        vd.metadata["shakedown_frame_labels_by_field"] = existing
    if details_by_field:
        existing = vd.metadata.get("shakedown_frame_details_by_field", {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update(details_by_field)
        vd.metadata["shakedown_frame_details_by_field"] = existing


def _shakedown_result_parts(raw: dict) -> list[dict[str, Any]]:
    records = _resultset_records(raw.get("ResultsSet"))
    if not records:
        return [_shakedown_arrays(raw)]
    return [_shakedown_arrays(raw, record) for record in records]


def _register_shakedown_fields(raw: dict, vd: VizData, selected_fields: Optional[set[str]]) -> None:
    """Derive visual fields from shakedown solver result MATs."""
    selected = set(_SHAKEDOWN_FIELD_SPECS) if selected_fields is None else selected_fields
    requested = set(_SHAKEDOWN_FIELD_SPECS).intersection(selected)
    if not requested:
        return

    result_parts = _shakedown_result_parts(raw)
    result_parts = [parts for parts in result_parts if _shakedown_counts(parts)[0] > 0]
    if not result_parts:
        return

    n_gauss, n_str, n_vert = _shakedown_counts(result_parts[0])
    if n_gauss <= 0:
        return

    n_results = len(result_parts)
    labels_by_field: dict[str, list[str]] = {}
    details_by_field: dict[str, list[dict[str, Any]]] = {}

    residual_frames: list[np.ndarray] = []
    residual_labels: list[str] = []
    residual_details: list[dict[str, Any]] = []
    total_frames: list[np.ndarray] = []
    total_labels: list[str] = []
    total_details: list[dict[str, Any]] = []
    ineq_frames: list[np.ndarray] = []
    ineq_labels: list[str] = []
    ineq_details: list[dict[str, Any]] = []
    multiplier_frames: list[np.ndarray] = []
    multiplier_labels: list[str] = []
    multiplier_details: list[dict[str, Any]] = []
    layer_residual_frames: dict[str, list[np.ndarray]] = {"SNEG": [], "SPOS": []}
    layer_residual_labels: dict[str, list[str]] = {"SNEG": [], "SPOS": []}
    layer_residual_details: dict[str, list[dict[str, Any]]] = {"SNEG": [], "SPOS": []}
    layer_total_frames: dict[str, list[np.ndarray]] = {"SNEG": [], "SPOS": []}
    layer_total_labels: dict[str, list[str]] = {"SNEG": [], "SPOS": []}
    layer_total_details: dict[str, list[dict[str, Any]]] = {"SNEG": [], "SPOS": []}
    generalized_residual_frames: dict[str, list[np.ndarray]] = {"SF": [], "SM": []}
    generalized_residual_labels: dict[str, list[str]] = {"SF": [], "SM": []}
    generalized_residual_details: dict[str, list[dict[str, Any]]] = {"SF": [], "SM": []}
    generalized_total_frames: dict[str, list[np.ndarray]] = {"SF": [], "SM": []}
    generalized_total_labels: dict[str, list[str]] = {"SF": [], "SM": []}
    generalized_total_details: dict[str, list[dict[str, Any]]] = {"SF": [], "SM": []}
    phi_frames: list[np.ndarray] = []
    phi_labels: list[str] = []
    phi_details: list[dict[str, Any]] = []
    active_branch_frames: list[np.ndarray] = []
    active_branch_labels: list[str] = []
    active_branch_details: list[dict[str, Any]] = []
    eq_frames: list[np.ndarray] = []
    eq_labels: list[str] = []
    eq_details: list[dict[str, Any]] = []
    activity_frames: list[np.ndarray] = []
    activity_labels: list[str] = []
    activity_details: list[dict[str, Any]] = []

    for idx, parts in enumerate(result_parts):
        _, _, row_n_vert = _shakedown_counts(parts)
        row_label = _format_shakedown_result_label(parts, idx, n_results)
        row_detail = _shakedown_result_detail(parts, idx, n_results)
        residual = _as_float_array(parts.get("residual_stress"))
        if residual is not None and residual.ndim == 2 and residual.shape == (n_gauss, n_str):
            residual_frames.append(residual)
            residual_labels.append(row_label)
            residual_details.append(_shakedown_frame_detail(row_detail))

            stress_cases = _shakedown_stress_cases(parts)
            vertex_load = _shakedown_vertex_load_matrix(parts)
            alpha = _alpha_scalar(parts.get("objective_value"))
            if (
                stress_cases is not None
                and vertex_load is not None
                and alpha is not None
                and stress_cases.ndim == 3
                and vertex_load.ndim == 2
                and stress_cases.shape[0] == n_gauss
                and stress_cases.shape[1] == n_str
                and stress_cases.shape[2] == vertex_load.shape[0]
            ):
                elastic_vertices = np.einsum("gsi,iv->gsv", stress_cases, vertex_load)
                total = residual[np.newaxis, :, :] + alpha * np.moveaxis(elastic_vertices, 2, 0)
                display_order = _shakedown_vertex_display_order(parts, int(total.shape[0]))
                for display_idx, frame_idx in enumerate(display_order):
                    vertex_detail = _shakedown_vertex_detail(
                        parts, frame_idx, int(total.shape[0]), display_index=display_idx
                    )
                    vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                        parts, idx, n_results, vertex_detail
                    )
                    total_frames.append(total[frame_idx])
                    vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{total.shape[0]}")
                    total_labels.append(f"{vertex_row_label} - {vertex_label}")
                    total_details.append(_shakedown_frame_detail(
                        vertex_row_detail,
                        vertex_detail=vertex_detail,
                    ))

        for surface in ("SNEG", "SPOS"):
            residual_surface = _shakedown_layer_residual_surface(parts, surface)
            if residual_surface is not None:
                layer_residual_frames[surface].append(residual_surface)
                layer_residual_labels[surface].append(row_label)
                layer_residual_details[surface].append(_shakedown_frame_detail(row_detail))
            total_surface = _shakedown_layer_total_surface(parts, surface)
            if total_surface is not None:
                display_order = _shakedown_vertex_display_order(parts, int(total_surface.shape[0]))
                for display_idx, frame_idx in enumerate(display_order):
                    vertex_detail = _shakedown_vertex_detail(
                        parts, frame_idx, int(total_surface.shape[0]), display_index=display_idx
                    )
                    vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                        parts, idx, n_results, vertex_detail
                    )
                    layer_total_frames[surface].append(total_surface[frame_idx])
                    vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{total_surface.shape[0]}")
                    layer_total_labels[surface].append(f"{vertex_row_label} - {vertex_label}")
                    layer_total_details[surface].append(_shakedown_frame_detail(
                        vertex_row_detail,
                        vertex_detail=vertex_detail,
                    ))

        for suffix in ("SF", "SM"):
            residual_gen = _shakedown_generalized_component(parts, total=False, suffix=suffix)
            if residual_gen is not None:
                generalized_residual_frames[suffix].append(residual_gen)
                generalized_residual_labels[suffix].append(row_label)
                generalized_residual_details[suffix].append(_shakedown_frame_detail(row_detail))
            total_gen = _shakedown_generalized_component(parts, total=True, suffix=suffix)
            if total_gen is not None:
                display_order = _shakedown_vertex_display_order(parts, int(total_gen.shape[0]))
                for display_idx, frame_idx in enumerate(display_order):
                    vertex_detail = _shakedown_vertex_detail(
                        parts, frame_idx, int(total_gen.shape[0]), display_index=display_idx
                    )
                    vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                        parts, idx, n_results, vertex_detail
                    )
                    generalized_total_frames[suffix].append(total_gen[frame_idx])
                    vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{total_gen.shape[0]}")
                    generalized_total_labels[suffix].append(f"{vertex_row_label} - {vertex_label}")
                    generalized_total_details[suffix].append(_shakedown_frame_detail(
                        vertex_row_detail,
                        vertex_detail=vertex_detail,
                    ))

        ratio_frames = _shakedown_yield_ratio_frames(
            parts,
            n_gauss if _shakedown_is_shell_layer(parts) else None,
        )
        if ratio_frames is not None:
            ineq = np.maximum(ratio_frames - 1.0, 0.0)
            display_order = _shakedown_vertex_display_order(parts, int(ineq.shape[0]))
            for display_idx, frame_idx in enumerate(display_order):
                vertex_detail = _shakedown_vertex_detail(
                    parts, frame_idx, int(ineq.shape[0]), display_index=display_idx
                )
                vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                    parts, idx, n_results, vertex_detail
                )
                vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{ineq.shape[0]}")
                detail = _shakedown_frame_detail(
                    vertex_row_detail,
                    vertex_detail=vertex_detail,
                )
                ineq_frames.append(ineq[frame_idx])
                ineq_labels.append(f"{vertex_row_label} - {vertex_label}")
                ineq_details.append(detail)
                if _shakedown_is_shell_ilyushin(parts):
                    phi_frames.append(ratio_frames[frame_idx])
                    phi_labels.append(f"{vertex_row_label} - {vertex_label}")
                    phi_details.append(detail)
                    active_branch_frames.append((ratio_frames[frame_idx] >= 0.999).astype(np.float64))
                    active_branch_labels.append(f"{vertex_row_label} - {vertex_label}")
                    active_branch_details.append(detail)

        multiplier = _shakedown_inequality_multiplier_frames(parts, n_gauss, row_n_vert)
        if multiplier is not None:
            display_order = _shakedown_vertex_display_order(parts, int(multiplier.shape[0]))
            for display_idx, frame_idx in enumerate(display_order):
                vertex_detail = _shakedown_vertex_detail(
                    parts, frame_idx, int(multiplier.shape[0]), display_index=display_idx
                )
                vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                    parts, idx, n_results, vertex_detail
                )
                multiplier_frames.append(multiplier[frame_idx])
                vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{multiplier.shape[0]}")
                multiplier_labels.append(f"{vertex_row_label} - {vertex_label}")
                multiplier_details.append(_shakedown_frame_detail(
                    vertex_row_detail,
                    vertex_detail=vertex_detail,
                ))

        activity = _as_float_array(parts.get("max_effective_excess_per_gp"))
        if activity is not None:
            values = activity.reshape(-1)
            if values.shape[0] == n_gauss:
                activity_frames.append(values)
                activity_labels.append(row_label)
                activity_details.append(_shakedown_frame_detail(row_detail))

        eq = _shakedown_equilibrium_node_field(raw, vd, row_n_vert, parts)
        if eq is not None:
            eq_frames.append(eq)
            eq_labels.append(row_label)
            eq_details.append(_shakedown_frame_detail(row_detail))

    if "shakedown_residual_stress" in requested and residual_frames:
        if len(residual_frames) == 1:
            residual_data = residual_frames[0]
            residual_n_frames = 1
        else:
            residual_data = np.stack(residual_frames, axis=0)
            residual_n_frames = int(residual_data.shape[0])
            labels_by_field["shakedown_residual_stress"] = residual_labels
            details_by_field["shakedown_residual_stress"] = residual_details
        vd.fields["shakedown_residual_stress"] = residual_data
        vd.field_info["shakedown_residual_stress"] = FieldInfo(
            key="shakedown_residual_stress",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_residual_stress"]["label"],
            n_components=n_str,
            location="gauss",
            n_frames=residual_n_frames,
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_residual_stress"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_residual_stress"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_residual_stress"].get("description", ""),
        )
        if residual_n_frames == 1 and residual_details:
            details_by_field["shakedown_residual_stress"] = residual_details

    if "shakedown_total_stress" in requested and total_frames:
        total_data = np.stack(total_frames, axis=0)
        vd.fields["shakedown_total_stress"] = total_data
        vd.field_info["shakedown_total_stress"] = FieldInfo(
            key="shakedown_total_stress",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_total_stress"]["label"],
            n_components=n_str,
            location="gauss",
            n_frames=int(total_data.shape[0]),
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_total_stress"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_total_stress"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_total_stress"].get("description", ""),
        )
        labels_by_field["shakedown_total_stress"] = total_labels
        details_by_field["shakedown_total_stress"] = total_details

    for surface in ("SNEG", "SPOS"):
        for key, force_frames in (
            (f"shakedown_S_res_{surface}", True),
            (f"shakedown_layer_residual_stress_{surface}", False),
        ):
            if key not in requested or not layer_residual_frames[surface]:
                continue
            if force_frames or len(layer_residual_frames[surface]) > 1:
                data = np.stack(layer_residual_frames[surface], axis=0)
                n_frames = int(data.shape[0])
                labels_by_field[key] = layer_residual_labels[surface]
                details_by_field[key] = layer_residual_details[surface]
            else:
                data = layer_residual_frames[surface][0]
                n_frames = 1
            vd.fields[key] = data
            vd.field_info[key] = FieldInfo(
                key=key,
                label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                n_components=3,
                location="gauss",
                n_frames=n_frames,
                symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
            )
            if (not force_frames) and n_frames == 1 and layer_residual_details[surface]:
                details_by_field[key] = layer_residual_details[surface]

        for key in (f"shakedown_S_tot_{surface}", f"shakedown_layer_total_stress_{surface}"):
            if key in requested and layer_total_frames[surface]:
                data = np.stack(layer_total_frames[surface], axis=0)
                vd.fields[key] = data
                vd.field_info[key] = FieldInfo(
                    key=key,
                    label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                    n_components=3,
                    location="gauss",
                    n_frames=int(data.shape[0]),
                    symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                    formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                    description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
                )
                labels_by_field[key] = layer_total_labels[surface]
                details_by_field[key] = layer_total_details[surface]

        key = f"shakedown_CInEQ_{surface}"
        if key in requested:
            surface_frames: list[np.ndarray] = []
            surface_labels: list[str] = []
            surface_details: list[dict[str, Any]] = []
            for idx, parts in enumerate(result_parts):
                ratio_surface = _shakedown_layer_yield_surface(parts, surface)
                if ratio_surface is None:
                    continue
                display_order = _shakedown_vertex_display_order(parts, int(ratio_surface.shape[0]))
                for display_idx, frame_idx in enumerate(display_order):
                    vertex_detail = _shakedown_vertex_detail(
                        parts, frame_idx, int(ratio_surface.shape[0]), display_index=display_idx
                    )
                    vertex_row_label, vertex_row_detail = _shakedown_vertex_result_label_and_detail(
                        parts, idx, n_results, vertex_detail
                    )
                    surface_frames.append(np.maximum(ratio_surface[frame_idx] - 1.0, 0.0))
                    vertex_label = str(vertex_detail.get("full") or f"Vertex {display_idx + 1}/{ratio_surface.shape[0]}")
                    surface_labels.append(f"{vertex_row_label} - {vertex_label}")
                    surface_details.append(_shakedown_frame_detail(
                        vertex_row_detail,
                        vertex_detail=vertex_detail,
                    ))
            if surface_frames:
                data = np.stack(surface_frames, axis=0)
                vd.fields[key] = data
                vd.field_info[key] = FieldInfo(
                    key=key,
                    label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                    n_components=1,
                    location="gauss",
                    n_frames=int(data.shape[0]),
                    symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                    formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                    description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
                )
                labels_by_field[key] = surface_labels
                details_by_field[key] = surface_details

    for suffix in ("SF", "SM"):
        for key, force_frames in (
            (f"shakedown_{suffix}_res", True),
            (f"shakedown_generalized_residual_{suffix}", False),
        ):
            if key not in requested or not generalized_residual_frames[suffix]:
                continue
            if force_frames or len(generalized_residual_frames[suffix]) > 1:
                data = np.stack(generalized_residual_frames[suffix], axis=0)
                n_frames = int(data.shape[0])
                labels_by_field[key] = generalized_residual_labels[suffix]
                details_by_field[key] = generalized_residual_details[suffix]
            else:
                data = generalized_residual_frames[suffix][0]
                n_frames = 1
            vd.fields[key] = data
            vd.field_info[key] = FieldInfo(
                key=key,
                label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                n_components=3,
                location="gauss",
                n_frames=n_frames,
                symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
            )
            if (not force_frames) and n_frames == 1 and generalized_residual_details[suffix]:
                details_by_field[key] = generalized_residual_details[suffix]

        for key in (f"shakedown_{suffix}_tot", f"shakedown_generalized_total_{suffix}"):
            if key in requested and generalized_total_frames[suffix]:
                data = np.stack(generalized_total_frames[suffix], axis=0)
                vd.fields[key] = data
                vd.field_info[key] = FieldInfo(
                    key=key,
                    label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                    n_components=3,
                    location="gauss",
                    n_frames=int(data.shape[0]),
                    symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                    formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                    description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
                )
                labels_by_field[key] = generalized_total_labels[suffix]
                details_by_field[key] = generalized_total_details[suffix]

    if "shakedown_inequality_violation" in requested and ineq_frames:
        ineq_data = np.stack(ineq_frames, axis=0)
        vd.fields["shakedown_inequality_violation"] = ineq_data
        vd.field_info["shakedown_inequality_violation"] = FieldInfo(
            key="shakedown_inequality_violation",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_violation"]["label"],
            n_components=1,
            location="gauss",
            n_frames=int(ineq_data.shape[0]),
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_violation"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_violation"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_violation"].get("description", ""),
        )
        labels_by_field["shakedown_inequality_violation"] = ineq_labels
        details_by_field["shakedown_inequality_violation"] = ineq_details

    for key in ("shakedown_Phi", "shakedown_ilyushin_phi"):
        if key in requested and phi_frames:
            phi_data = np.stack(phi_frames, axis=0)
            vd.fields[key] = phi_data
            vd.field_info[key] = FieldInfo(
                key=key,
                label=_SHAKEDOWN_FIELD_SPECS[key]["label"],
                n_components=1,
                location="gauss",
                n_frames=int(phi_data.shape[0]),
                symbol=_SHAKEDOWN_FIELD_SPECS[key].get("symbol", ""),
                formula=_SHAKEDOWN_FIELD_SPECS[key].get("formula", ""),
                description=_SHAKEDOWN_FIELD_SPECS[key].get("description", ""),
            )
            labels_by_field[key] = phi_labels
            details_by_field[key] = phi_details

    if "shakedown_CInEQ" in requested and ineq_frames:
        cineq_data = np.stack(ineq_frames, axis=0)
        vd.fields["shakedown_CInEQ"] = cineq_data
        vd.field_info["shakedown_CInEQ"] = FieldInfo(
            key="shakedown_CInEQ",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_CInEQ"]["label"],
            n_components=1,
            location="gauss",
            n_frames=int(cineq_data.shape[0]),
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_CInEQ"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_CInEQ"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_CInEQ"].get("description", ""),
        )
        labels_by_field["shakedown_CInEQ"] = ineq_labels
        details_by_field["shakedown_CInEQ"] = ineq_details

    if "shakedown_ilyushin_active_branch" in requested and active_branch_frames:
        active_data = np.stack(active_branch_frames, axis=0)
        vd.fields["shakedown_ilyushin_active_branch"] = active_data
        vd.field_info["shakedown_ilyushin_active_branch"] = FieldInfo(
            key="shakedown_ilyushin_active_branch",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_ilyushin_active_branch"]["label"],
            n_components=1,
            location="gauss",
            n_frames=int(active_data.shape[0]),
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_ilyushin_active_branch"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_ilyushin_active_branch"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_ilyushin_active_branch"].get("description", ""),
        )
        labels_by_field["shakedown_ilyushin_active_branch"] = active_branch_labels
        details_by_field["shakedown_ilyushin_active_branch"] = active_branch_details

    if "shakedown_inequality_multiplier" in requested and multiplier_frames:
        multiplier_data = np.stack(multiplier_frames, axis=0)
        vd.fields["shakedown_inequality_multiplier"] = multiplier_data
        vd.field_info["shakedown_inequality_multiplier"] = FieldInfo(
            key="shakedown_inequality_multiplier",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_multiplier"]["label"],
            n_components=1,
            location="gauss",
            n_frames=int(multiplier_data.shape[0]),
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_multiplier"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_multiplier"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_inequality_multiplier"].get("description", ""),
        )
        labels_by_field["shakedown_inequality_multiplier"] = multiplier_labels
        details_by_field["shakedown_inequality_multiplier"] = multiplier_details

    if "max_effective_excess_per_gp" in requested and activity_frames:
        if len(activity_frames) == 1:
            activity_data = activity_frames[0]
            activity_n_frames = 1
        else:
            activity_data = np.stack(activity_frames, axis=0)
            activity_n_frames = int(activity_data.shape[0])
            labels_by_field["max_effective_excess_per_gp"] = activity_labels
            details_by_field["max_effective_excess_per_gp"] = activity_details
        vd.fields["max_effective_excess_per_gp"] = activity_data
        vd.field_info["max_effective_excess_per_gp"] = FieldInfo(
            key="max_effective_excess_per_gp",
            label=_SHAKEDOWN_FIELD_SPECS["max_effective_excess_per_gp"]["label"],
            n_components=1,
            location="gauss",
            n_frames=activity_n_frames,
            symbol=_SHAKEDOWN_FIELD_SPECS["max_effective_excess_per_gp"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["max_effective_excess_per_gp"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["max_effective_excess_per_gp"].get("description", ""),
        )
        if activity_n_frames == 1 and activity_details:
            details_by_field["max_effective_excess_per_gp"] = activity_details

    if "shakedown_equality_violation" in requested and eq_frames:
        if len(eq_frames) == 1:
            eq_data = eq_frames[0]
            eq_n_frames = 1
        else:
            eq_data = np.stack(eq_frames, axis=0)
            eq_n_frames = int(eq_data.shape[0])
            labels_by_field["shakedown_equality_violation"] = eq_labels
            details_by_field["shakedown_equality_violation"] = eq_details
        vd.fields["shakedown_equality_violation"] = eq_data
        vd.field_info["shakedown_equality_violation"] = FieldInfo(
            key="shakedown_equality_violation",
            label=_SHAKEDOWN_FIELD_SPECS["shakedown_equality_violation"]["label"],
            n_components=int(eq_data.shape[-1]) if eq_data.ndim >= 2 else 1,
            location="node",
            n_frames=eq_n_frames,
            symbol=_SHAKEDOWN_FIELD_SPECS["shakedown_equality_violation"].get("symbol", ""),
            formula=_SHAKEDOWN_FIELD_SPECS["shakedown_equality_violation"].get("formula", ""),
            description=_SHAKEDOWN_FIELD_SPECS["shakedown_equality_violation"].get("description", ""),
        )
        if eq_n_frames == 1 and eq_details:
            details_by_field["shakedown_equality_violation"] = eq_details

    if labels_by_field:
        vd.metadata["shakedown_frame_labels_by_field"] = labels_by_field
    if details_by_field:
        vd.metadata["shakedown_frame_details_by_field"] = details_by_field


def _enrich_shakedown_metadata(raw: dict, vd: VizData) -> None:
    """Attach lightweight shakedown frame labels used by the frontend."""
    parts = _shakedown_arrays(raw)
    _, _, n_vert = _shakedown_counts(parts)
    labels = _shakedown_vertex_labels(parts, n_vert)
    if labels:
        vd.metadata["shakedown_vertex_labels"] = labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_viz_data(
    mat_path: str | Path,
    selected_fields: Optional[Sequence[str]] = None,
    *,
    validation_side: Optional[str] = None,
) -> VizData:
    """Load a .mat file and return a ``VizData`` ready for visualisation.

    Parameters
    ----------
    mat_path : str or Path
        Path to a jaxmech .mat file.

    Returns
    -------
    VizData
        Unified data container with mesh and discovered field variables.
    """
    import scipy.io as sio

    mat_path = Path(mat_path)
    selected_set = {str(k) for k in selected_fields or [] if str(k).strip()}
    variable_names = None
    validation_side_key = str(validation_side or "").strip().lower()
    if validation_side_key not in {"jax", "abaqus"}:
        validation_side_key = ""
    if validation_side_key:
        variable_names = sorted(_validation_required_raw_keys(selected_set, validation_side_key))
    elif selected_set:
        dependency_keys = _selected_field_dependencies(selected_set)
        required = {
            "InpData",
            FRAME_OUTPUTS_KEY,
            SHELL_PARAM_KEY,
            SOLID_PARAM_KEY,
            VIZ_MANIFEST_KEY,
            "metadata",
            "n_gauss_per_elem",
            "frame_time",
            "time_grid",
            "frame_load_scale",
            "viz_points",
            "viz_cells",
            "viz_cell_types",
            "viz_cell_block_ele_types",
        }
        if (
            any(key in _SHAKEDOWN_FIELD_SPECS for key in selected_set)
            or _selection_has_rsdms_fields(selected_set)
            or _selection_has_rsdm_fields(selected_set)
        ):
            required.update({
                "ElasticInputSet",
                "ResultsSet",
                "ConfigInfo",
                "RSDMResult",
                "RSDMInput",
                SHELL_PARAM_KEY,
                "gauss_coords",
                "gauss_vols",
                "gauss_stress_cases",
                "sigma_E",
                "vertex_load_matrix",
                "load_factor",
                "R_ratios",
                "theta_deg",
                "phi_deg",
                "load_factor_set",
                "objective_value",
                "residual_stress",
                "yield_ratio",
                "dual_variables_ineq",
                "max_effective_excess_per_gp",
                "family",
                "shell_yield",
                "ilyushin_c",
                "residual_stress_kind",
                "yield_measure",
                "shell_layer_stress_cases",
                "shell_layer_strain_cases",
                "shell_generalized_stress_cases",
                "shell_section_z",
                "shell_elem_thickness",
                "residual_generalized_stress",
                "free_dofs",
                "C_sparse",
                "equilibrium_residual",
                "equilibrium_residual_norm1",
                "NumVert",
                "RSDMShakedownTrace",
                "RSDMShakedownInput",
            })
        variable_names = sorted(selected_set | required | dependency_keys)
        if any(
            key.startswith("frame_") or key.startswith("abaqus_frame_")
            for key in selected_set | dependency_keys
        ):
            variable_names = sorted(set(variable_names) | {FRAME_OUTPUTS_KEY, SOLID_PARAM_KEY})
        if any(key.startswith("shell_") or key == "n_generalized_str" for key in selected_set | dependency_keys):
            variable_names = sorted(set(variable_names) | {SHELL_PARAM_KEY})
    raw_loaded = sio.loadmat(
        str(mat_path),
        squeeze_me=False,
        struct_as_record=True,
        variable_names=variable_names,
    )
    raw = expand_public_mat_payload(raw_loaded)
    manifest = extract_viz_manifest(raw)
    if manifest is None and selected_set:
        try:
            manifest = load_viz_manifest_from_mat(mat_path)
        except Exception:
            manifest = None
    if _raw_is_rsdm_shakedown(raw, manifest):
        manifest = _filter_rsdm_shakedown_manifest(manifest)
    manifest = _normalize_manifest_for_reader(manifest, raw.keys())
    if validation_side_key and _as_validation_flag(raw):
        manifest = None
    if manifest is not None:
        allowed_fields = {
            str(field.get("key"))
            for field in manifest.get("fields", [])
            if isinstance(field, Mapping) and str(field.get("key", "")).strip()
        }
        if str(manifest.get("analysis_type", "")).strip().lower() == "shakedown":
            allowed_fields.update(_SHAKEDOWN_FIELD_SPECS)
        if selected_set:
            selected_set = selected_set.intersection(allowed_fields)
        else:
            selected_set = set(allowed_fields)

    vd = VizData(source_path=str(mat_path))
    vd.metadata = _extract_metadata(raw)

    # --- mesh ---
    pts, cells, ctypes, eletypes = _extract_mesh_from_viz_keys(raw)
    if pts is None:
        pts, cells, ctypes, eletypes = _extract_mesh_from_inpdata(raw)
    if pts is None:
        pts, cells, ctypes, eletypes = _extract_mesh_from_model_fallback(mat_path, raw)
    if pts is None:
        pts, cells, ctypes, eletypes = _extract_mesh_from_gauss_points(raw)
    vd.points = pts
    vd.cells = cells
    vd.cell_types = ctypes
    vd.cell_ele_types = eletypes

    # --- n_gauss bookkeeping ---
    ngpe = raw.get("n_gauss_per_elem")
    if ngpe is not None:
        vd.n_gauss_per_elem = _squeeze(np.asarray(ngpe, dtype=np.int32))
    if vd.n_gauss_per_elem is None:
        vd.n_gauss_per_elem = _infer_n_gauss_per_elem_from_raw(raw)
    if vd.cell_ele_types is not None:
        try:
            if np.all(np.asarray(vd.cell_ele_types, dtype=object) == "GAUSS_POINT"):
                vd.n_gauss_per_elem = np.ones(int(vd.cell_types.shape[0]), dtype=np.int32)
        except Exception:
            pass

    # --- fields ---
    field_selection = selected_set if (selected_set or manifest is not None) else None
    if validation_side_key and _as_validation_flag(raw):
        _register_validation_compare_fields(
            raw,
            vd,
            field_selection,
            side=validation_side_key,
        )
    else:
        _discover_fields(raw, vd, field_selection)
        _register_selected_extra_fields(raw, vd, field_selection)
        _register_shell_incremental_derived_fields(raw, vd, field_selection)
        _register_shakedown_fields(raw, vd, field_selection)
        _register_rsdm_steady_fields(raw, vd, field_selection)
        _register_rsdms_summary_fields(raw, vd, field_selection)
        _register_rsdms_path_fields(raw, vd, field_selection)
        _apply_manifest_field_info(vd, manifest, field_selection)

    # --- frame times ---
    vd.frame_times = _extract_frame_times(raw)

    # --- metadata ---
    _enrich_shakedown_metadata(raw, vd)
    _enrich_rsdms_summary_frame_metadata(raw, vd)

    # --- derived fields (von Mises, displacement magnitude, etc.) ---
    if vd.has_mesh:
        from jaxmech.modules.visualization.derived import register_derived_fields
        register_derived_fields(vd)

    return vd
