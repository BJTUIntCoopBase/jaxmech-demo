#!/usr/bin/env python3
"""
validate_abaqus_odb.py
======================

Validate one solid elastic MAT payload against the corresponding ABAQUS ODB result.

Workflow:
1. Start from one saved JAX MAT payload generated after elastic analysis.
2. Extract or load one ABAQUS MAT derived from the ODB.
3. Compare U / S / E / NFORC directly against the saved JAX fields.
4. Optionally write validation flags back into the same MAT payload.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from jaxmech.env.env import PROJECT_ROOT
from jaxmech.env.model_paths import (
    normalize_jax_system,
    resolve_model_case_paths,
    win_to_wsl_path,
)

_EXTERNAL_ROOT = PROJECT_ROOT / "external"
for _path in (PROJECT_ROOT, _EXTERNAL_ROOT, _EXTERNAL_ROOT / "JaxSSO"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from jaxmech.io.abaqus.odb import extract_solid_odb
from jaxmech.io.abaqus.inp import parse_inp
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult
from jaxmech.modules.shakedown.config import parse_shakedown_config


@dataclass(frozen=True)
class _MatMeshBlock:
    ele_type: str
    connectivity: np.ndarray


@dataclass(frozen=True)
class _MatMeshView:
    points: np.ndarray
    blocks: tuple[_MatMeshBlock, ...]

    @property
    def coord_dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_elements(self) -> int:
        return int(sum(block.connectivity.shape[0] for block in self.blocks))


def _load_mat(mat_path: str | Path) -> dict:
    return sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)


def _save_mat(mat_path: str | Path, payload: dict) -> None:
    sio.savemat(str(mat_path), payload, do_compression=True)


def _clean_payload(raw: dict) -> dict:
    payload = {k: v for k, v in raw.items() if not k.startswith("__")}
    for key, value in list(payload.items()):
        unwrapped = _unwrap_scalar(value)
        if hasattr(unwrapped, "_fieldnames"):
            payload[key] = {name: getattr(unwrapped, name) for name in unwrapped._fieldnames}
    return payload


def _mat_struct_to_dict(obj) -> dict:
    obj = _unwrap_scalar(obj)
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        rec = obj[0, 0]
        return {name: rec[name] for name in obj.dtype.names}
    raise TypeError(f"Unsupported MATLAB struct object: {type(obj)!r}")


def _unwrap_scalar(value):
    cur = value
    while isinstance(cur, np.ndarray) and cur.dtype == object and cur.size == 1:
        cur = cur.reshape(-1)[0]
    return cur


def _as_str(value) -> str:
    return str(np.asarray(_unwrap_scalar(value)).reshape(-1)[0])


def _as_optional_str(value, default: str = "") -> str:
    cur = _unwrap_scalar(value)
    if cur is None:
        return default
    arr = np.asarray(cur)
    if arr.size == 0:
        return default
    text = str(arr.reshape(-1)[0]).strip()
    return text or default


def _as_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _infer_odb_path_from_mat(raw: dict) -> str:
    inp_struct = _unwrap_scalar(raw["InpData"])
    inp_path = _as_str(inp_struct.inp_path)
    if not inp_path.lower().endswith(".inp"):
        raise ValueError(f"Cannot infer ODB path from inp_path: {inp_path}")
    return inp_path[:-4] + ".odb"


def _to_windows_path_from_wsl(path_obj: str | Path) -> str:
    path = str(path_obj)
    if path.startswith("/mnt/") and len(path) > 6:
        drive = path[5].upper()
        tail = path[7:].replace("/", "\\")
        return f"{drive}:\\{tail}"
    return path.replace("/", "\\")


def _ensure_windows_odb_from_inp(config: dict, case_name: str, inp_wsl_path: str | Path) -> tuple[str, str]:
    """Resolve or generate the target ODB on the Windows-side model tree.

    Search order:
    1. <win_root>/<model_name>/<case>.odb
    2. <win_root>/<model_name>/abaqus/<case>.odb
    3. <win_root>/<model_name>/ABAQUS/<case>.odb
    4. <win_root>/<model_name>/<case>/<case>.odb
    5. Otherwise create <win_root>/<model_name>/<case>/, copy the INP there,
       run ABAQUS, and use the generated ODB.

    Returns
    -------
    tuple[str, str]
        (job_dir_wsl, odb_name)
    """
    win_root_wsl = win_to_wsl_path(config["win_stored_root"])
    model_dir = win_root_wsl / str(config["model_name"])
    model_dir.mkdir(parents=True, exist_ok=True)

    root_candidates = [
        model_dir / f"{case_name}.odb",
        model_dir / "abaqus" / f"{case_name}.odb",
        model_dir / "ABAQUS" / f"{case_name}.odb",
    ]
    for candidate in root_candidates:
        if candidate.is_file():
            return str(candidate.parent), candidate.name

    case_dir = model_dir / case_name
    case_odb = case_dir / f"{case_name}.odb"
    if case_odb.is_file():
        return str(case_dir), case_odb.name

    case_dir.mkdir(parents=True, exist_ok=True)
    inp_src = Path(inp_wsl_path)
    if not inp_src.is_file():
        raise FileNotFoundError(f"INP file not found for ODB generation: {inp_src}")
    inp_dst = case_dir / f"{case_name}.inp"
    if not inp_dst.exists():
        shutil.copy2(inp_src, inp_dst)

    win_case_dir = _to_windows_path_from_wsl(case_dir)
    abaqus_cmd = str(config.get("windows_abaqus_cmd", "abaqus")).strip() or "abaqus"
    cmd = f'cmd.exe /c "cd /d {win_case_dir} && {abaqus_cmd} job={case_name} input={case_name}.inp"'
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ABAQUS job submission failed for {case_name} in {case_dir} "
            f"(exit code {result.returncode})."
        )
    if not case_odb.is_file():
        raise FileNotFoundError(f"Expected generated ODB not found: {case_odb}")
    return str(case_dir), case_odb.name


def _legacy_nodal_disp(ele_node_label_set, ele_node_disp_set) -> np.ndarray:
    node_map = {}
    labels_arr = np.asarray(ele_node_label_set, dtype=object)
    disp_arr = np.asarray(ele_node_disp_set, dtype=object)
    if (labels_arr.ndim >= 2 and disp_arr.ndim >= 2
            and labels_arr.shape[0] == disp_arr.shape[0]
            and labels_arr.shape[0] > 1):
        pairs = [(labels_arr[i], disp_arr[i]) for i in range(labels_arr.shape[0])]
    else:
        pairs = zip(labels_arr.reshape(-1), disp_arr.reshape(-1))
    for labels_entry, disp_entry in pairs:
        labels = _mat_obj_numeric_array(labels_entry, dtype=np.int32).reshape(-1)
        disp = _mat_obj_numeric_array(disp_entry, dtype=np.float64).reshape(len(labels), -1)
        for lab, vec in zip(labels.tolist(), disp):
            if int(lab) not in node_map:
                node_map[int(lab)] = np.asarray(vec, dtype=np.float64).reshape(-1)
    ordered_labels = np.array(sorted(node_map), dtype=np.int32)
    return np.vstack([node_map[int(lab)] for lab in ordered_labels])


def _jax_nodal_disp(inp_struct, ele_node_disp_set_jax) -> np.ndarray:
    points = np.asarray(inp_struct.points, dtype=np.float64)
    cell_type_names = [_as_str(x) for x in inp_struct.cell_type_names.reshape(-1)]
    cell_type_cells = inp_struct.cell_type_cells.reshape(-1)
    ele_disp = np.asarray(ele_node_disp_set_jax, dtype=np.float64)
    nodal = np.zeros((points.shape[0], ele_disp.shape[2]), dtype=np.float64)
    assigned = np.zeros((points.shape[0],), dtype=bool)
    for idx in range(len(cell_type_names)):
        cells = np.asarray(cell_type_cells[idx], dtype=np.int32)
        if cells.shape[0] != ele_disp.shape[0]:
            continue
        for e in range(cells.shape[0]):
            conn = cells[e]
            nodal[conn, :] = ele_disp[e]
            assigned[conn] = True
        break
    if not np.all(assigned):
        missing = np.where(~assigned)[0][:10].tolist()
        raise ValueError(f"Some nodes were not assigned JAX displacement values: {missing}")
    return nodal


def _coord_key(coord: np.ndarray, decimals: int = 7) -> tuple[float, ...]:
    arr = np.asarray(coord, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr, decimals=decimals).tolist())


def _mat_obj_numeric_array(value, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(_unwrap_scalar(value))
    if arr.dtype != object:
        return np.asarray(arr, dtype=dtype)
    flat = [np.asarray(_unwrap_scalar(x)).reshape(-1)[0] for x in arr.reshape(-1)]
    return np.asarray(flat, dtype=dtype).reshape(arr.shape)


def _abaqus_node_data_from_legacy_fields(aba_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    label_to_coord: dict[int, np.ndarray] = {}
    label_set = np.asarray(aba_raw["EleNodeLabelSet"], dtype=object)
    coord_set = np.asarray(aba_raw["EleNodeCoordSet"], dtype=object)
    if (label_set.ndim >= 2 and coord_set.ndim >= 2
            and label_set.shape[0] == coord_set.shape[0]
            and label_set.shape[0] > 1):
        pairs = [(label_set[i], coord_set[i]) for i in range(label_set.shape[0])]
    else:
        pairs = zip(label_set.reshape(-1), coord_set.reshape(-1))
    for labels_obj, coords_obj in pairs:
        labels = _mat_obj_numeric_array(labels_obj, dtype=np.int32).reshape(-1)
        coords = _mat_obj_numeric_array(coords_obj, dtype=np.float64).reshape(len(labels), -1)
        for lab, coord in zip(labels.tolist(), coords):
            label_to_coord.setdefault(int(lab), np.asarray(coord, dtype=np.float64))

    ordered_labels = np.asarray(sorted(label_to_coord), dtype=np.int32)
    ordered_coords = np.vstack([label_to_coord[int(lab)] for lab in ordered_labels])
    return ordered_labels, ordered_coords


def _abaqus_element_topology_from_raw(aba_raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ele_types = np.asarray(
        [str(np.asarray(_unwrap_scalar(x)).reshape(-1)[0]).upper() for x in aba_raw["EleTypeSet"].reshape(-1)],
        dtype=object,
    )
    ele_labels = np.asarray(
        [int(np.asarray(_unwrap_scalar(x)).reshape(-1)[0]) for x in aba_raw["EleLabelSet"].reshape(-1)],
        dtype=np.int32,
    )
    conn_raw = np.asarray(aba_raw["EleNodeLabelSet"], dtype=object)
    if conn_raw.ndim >= 2 and conn_raw.shape[0] == ele_labels.size:
        conn_rows = [conn_raw[i] for i in range(conn_raw.shape[0])]
    else:
        conn_rows = conn_raw.reshape(-1)

    elem_conn = np.empty((len(conn_rows),), dtype=object)
    for i, row in enumerate(conn_rows):
        arr = np.asarray(row, dtype=object)
        flat = [int(np.asarray(_unwrap_scalar(x)).reshape(-1)[0]) for x in arr.reshape(-1)]
        elem_conn[i] = np.asarray(flat, dtype=np.int32)
    return ele_labels, elem_conn, ele_types


def _jax_nodal_data(jax_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    inp_struct = _unwrap_scalar(jax_raw["InpData"])
    points = np.asarray(inp_struct.points, dtype=np.float64)
    if "NodalDisp_jax" in jax_raw:
        disp = np.asarray(jax_raw["NodalDisp_jax"], dtype=np.float64)
    else:
        disp = _jax_nodal_disp(inp_struct, jax_raw["EleNodeDispSet_jax"])
    return points, disp


def _infer_jax_vec_dim(jax_raw: dict) -> tuple[int, int]:
    inp_struct = _unwrap_scalar(jax_raw["InpData"])
    points = np.asarray(inp_struct.points, dtype=np.float64)
    if "NodalDisp_jax" in jax_raw:
        disp = np.asarray(jax_raw["NodalDisp_jax"], dtype=np.float64)
        return int(disp.shape[1]), int(points.shape[1])
    if "EleNodeDispSet_jax" in jax_raw:
        ele_disp = np.asarray(jax_raw["EleNodeDispSet_jax"], dtype=np.float64)
        return int(ele_disp.shape[-1]), int(points.shape[1])
    if "EleNodeDispSetByType_jax" in jax_raw:
        obj = jax_raw["EleNodeDispSetByType_jax"].reshape(-1)[0]
        ele_disp = np.asarray(obj, dtype=np.float64)
        return int(ele_disp.shape[-1]), int(points.shape[1])
    return int(points.shape[1]), int(points.shape[1])


def _match_trailing_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-1] == target_dim:
        return arr
    if arr.shape[-1] > target_dim:
        return arr[..., :target_dim]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target_dim - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def _reorder_jax_nodal_to_abaqus(aba_coords: np.ndarray, jax_coords: np.ndarray, jax_values: np.ndarray) -> np.ndarray:
    cmp_dim = min(int(np.asarray(aba_coords).shape[1]), int(np.asarray(jax_coords).shape[1]))
    aba_cmp = np.asarray(aba_coords, dtype=np.float64)[:, :cmp_dim]
    jax_cmp = np.asarray(jax_coords, dtype=np.float64)[:, :cmp_dim]
    coord_to_jax = {}
    for idx, coord in enumerate(jax_cmp):
        coord_to_jax[_coord_key(coord)] = idx
    mapped = np.zeros((aba_coords.shape[0], jax_values.shape[1]), dtype=np.float64)
    for i, coord in enumerate(aba_cmp):
        key = _coord_key(coord)
        j = coord_to_jax.get(key)
        if j is None:
            d = np.linalg.norm(jax_cmp - coord[None, :], axis=1)
            j = int(np.argmin(d))
            if float(d[j]) > 1e-6:
                raise KeyError(f"Cannot map ABAQUS node coordinate to JAX node: {coord.tolist()}")
        mapped[i, :] = jax_values[j, :]
    return mapped


SUPPORTED_SOLID_ELE_TYPES = {"C3D8", "C3D8R", "C3D6", "C3D4", "CPS4", "CPS4R", "CPS3", "CPE4"}
DEFAULT_ELE_TYPE_FROM_CELL_TYPE = {
    "quad": "CPS4",
    "quadrilateral": "CPS4",
    "triangle": "CPS3",
    "hexahedron": "C3D8",
    "wedge": "C3D6",
    "tetra": "C3D4",
}
_CPS4_LEGACY_QP_SEQUENCE = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _reorder_block_gauss_to_legacy(ele_type: str, field: np.ndarray) -> np.ndarray:
    """Match the historical MAT Gauss-point ordering used by ABAQUS validation.

    The native solid recovery path already matches the ABAQUS ODB ordering for
    current 3D element support. Only the legacy 2D full-integration quad path
    still needs an explicit reorder here.
    """
    arr = np.asarray(field, dtype=np.float64)
    ele = str(ele_type).upper()
    if arr.ndim < 3:
        return arr
    if ele in {"CPS4", "CPE4"} and arr.shape[1] == 4:
        return arr[:, _CPS4_LEGACY_QP_SEQUENCE, ...]
    return arr


def _abaqus_element_blocks(aba_raw: dict) -> dict[str, dict[str, list[np.ndarray]]]:
    ele_types = [str(np.asarray(x).reshape(-1)[0]).upper() for x in aba_raw["EleTypeSet"].reshape(-1)]
    coord_raw = np.asarray(aba_raw["EleNodeCoordSet"], dtype=object)
    stress_raw = np.asarray(aba_raw["EleGaussStressSet"], dtype=object)
    strain_raw = np.asarray(aba_raw["EleGaussStrainSet"], dtype=object)
    if coord_raw.ndim >= 2 and stress_raw.ndim >= 2 and strain_raw.ndim >= 2 and coord_raw.shape[0] == len(ele_types):
        coord_objs = [coord_raw[i] for i in range(coord_raw.shape[0])]
        stress_objs = [stress_raw[i] for i in range(stress_raw.shape[0])]
        strain_objs = [strain_raw[i] for i in range(strain_raw.shape[0])]
    else:
        coord_objs = coord_raw.reshape(-1)
        stress_objs = stress_raw.reshape(-1)
        strain_objs = strain_raw.reshape(-1)
    blocks: dict[str, dict[str, list[np.ndarray]]] = {}
    for ele_type, coord_obj, stress_obj, strain_obj in zip(ele_types, coord_objs, stress_objs, strain_objs):
        if ele_type not in SUPPORTED_SOLID_ELE_TYPES:
            continue
        coords = _mat_obj_numeric_array(coord_obj, dtype=np.float64).reshape(np.asarray(coord_obj).shape[0], -1)
        stress = _mat_obj_numeric_array(stress_obj, dtype=np.float64).reshape(np.asarray(stress_obj).shape[0], -1)
        strain = _mat_obj_numeric_array(strain_obj, dtype=np.float64).reshape(np.asarray(strain_obj).shape[0], -1)
        block = blocks.setdefault(ele_type, {"coords": [], "stress": [], "strain": []})
        block["coords"].append(coords)
        block["stress"].append(stress)
        block["strain"].append(strain)
    return blocks


def _jax_element_blocks(jax_raw: dict) -> dict[str, dict[str, np.ndarray]]:
    inp_struct = _unwrap_scalar(jax_raw["InpData"])
    points = np.asarray(inp_struct.points, dtype=np.float64)
    if "mixed_element_jax" in jax_raw and _as_int(jax_raw["mixed_element_jax"]) == 1:
        if "cell_block_ele_types_jax" in jax_raw:
            ele_type_names = [str(np.asarray(x).reshape(-1)[0]).upper() for x in jax_raw["cell_block_ele_types_jax"].reshape(-1)]
            cell_objs = jax_raw["cell_block_cells_jax"].reshape(-1)
        else:
            type_names = [str(np.asarray(x).reshape(-1)[0]) for x in jax_raw["cell_type_names_jax"].reshape(-1)]
            ele_type_names = [DEFAULT_ELE_TYPE_FROM_CELL_TYPE.get(name, str(name).upper()) for name in type_names]
            cell_objs = jax_raw["cell_type_cells_jax"].reshape(-1)
        stress_objs = jax_raw["EleGaussStressSetByType_jax"].reshape(-1)
        strain_objs = jax_raw["EleGaussStrainSetByType_jax"].reshape(-1)
        blocks = {}
        for ele_type, cell_obj, stress_obj, strain_obj in zip(ele_type_names, cell_objs, stress_objs, strain_objs):
            cells = np.asarray(cell_obj, dtype=np.int32)
            block = blocks.setdefault(ele_type, {"coords": [], "stress": [], "strain": []})
            block["coords"].append(np.asarray(points[cells], dtype=np.float64))
            block["stress"].append(np.asarray(stress_obj, dtype=np.float64))
            block["strain"].append(np.asarray(strain_obj, dtype=np.float64))
        return {
            ele_type: {
                "coords": np.concatenate(data["coords"], axis=0),
                "stress": np.concatenate(data["stress"], axis=0),
                "strain": np.concatenate(data["strain"], axis=0),
            }
            for ele_type, data in blocks.items()
        }

    cell_type_names = [_as_str(x) for x in inp_struct.cell_type_names.reshape(-1)]
    cell_type_cells = inp_struct.cell_type_cells.reshape(-1)
    blocks = {}
    for cell_type_name, cell_obj in zip(cell_type_names, cell_type_cells):
        ele_type = DEFAULT_ELE_TYPE_FROM_CELL_TYPE.get(cell_type_name, str(cell_type_name).upper())
        cells = np.asarray(cell_obj, dtype=np.int32)
        blocks[ele_type] = {
            "coords": np.asarray(points[cells], dtype=np.float64),
            "stress": np.asarray(jax_raw["EleGaussStressSet_jax"], dtype=np.float64),
            "strain": np.asarray(jax_raw["EleGaussStrainSet_jax"], dtype=np.float64),
        }
    return blocks


def _element_signature(coords: np.ndarray) -> tuple[tuple[float, ...], ...]:
    keys = [_coord_key(row, decimals=6) for row in np.asarray(coords, dtype=np.float64)]
    return tuple(sorted(keys))


def _align_jax_block_to_abaqus(aba_coords_list: list[np.ndarray], jax_block: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    coords_jax = np.asarray(jax_block["coords"], dtype=np.float64)
    stress_jax = np.asarray(jax_block["stress"], dtype=np.float64)
    strain_jax = np.asarray(jax_block["strain"], dtype=np.float64)
    coord_dim = min(int(coords_jax.shape[-1]), int(np.asarray(aba_coords_list[0], dtype=np.float64).shape[-1]))
    coords_jax_cmp = coords_jax[..., :coord_dim]
    sig_to_idx = {}
    for i, coords in enumerate(coords_jax_cmp):
        sig = _element_signature(coords)
        if sig in sig_to_idx:
            raise ValueError("Duplicate element signature found in JAX block.")
        sig_to_idx[sig] = i

    stress_aligned = []
    strain_aligned = []
    remaining = set(range(coords_jax.shape[0]))
    centroids_jax = np.mean(coords_jax_cmp, axis=1)
    for coords in aba_coords_list:
        coords_cmp = np.asarray(coords, dtype=np.float64)[..., :coord_dim]
        sig = _element_signature(coords_cmp)
        j = sig_to_idx.get(sig)
        if j is None or j not in remaining:
            ctr = np.mean(coords_cmp, axis=0)
            cand = np.asarray(sorted(remaining), dtype=np.int32)
            d = np.linalg.norm(centroids_jax[cand] - ctr[None, :], axis=1)
            j = int(cand[int(np.argmin(d))])
            if float(np.min(d)) > 1e-6:
                raise KeyError("Cannot align ABAQUS element to JAX element by node coordinates.")
        remaining.discard(j)
        stress_aligned.append(stress_jax[j])
        strain_aligned.append(strain_jax[j])
    return np.asarray(stress_aligned, dtype=np.float64), np.asarray(strain_aligned, dtype=np.float64)


def _analysis_ndof_per_node(model: Model, result: AnalysisResult) -> int:
    return int(result.metadata.get("ndof_per_node", model.mesh.n_dim))


def _analysis_ndof_per_node_from_mat(mesh_view: _MatMeshView, result: AnalysisResult) -> int:
    metadata = dict(result.metadata or {})
    if "ndof_per_node" in metadata:
        try:
            return max(1, int(float(metadata.get("ndof_per_node"))))
        except Exception:
            pass

    for values in (result.u, result.internal_force):
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1 and mesh_view.n_nodes > 0 and arr.size % mesh_view.n_nodes == 0:
            return max(1, int(arr.size // mesh_view.n_nodes))
        if arr.ndim == 2 and arr.shape[0] == mesh_view.n_nodes:
            return max(1, int(arr.shape[1]))

    return mesh_view.coord_dim


def _reshape_nodal_field(values: np.ndarray | None, n_nodes: int, ndof_per_node: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"AnalysisResult.{name} is required for solid ODB validation.")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != n_nodes * ndof_per_node:
            raise ValueError(
                f"{name} size mismatch: got {arr.size}, expected {n_nodes * ndof_per_node}."
            )
        return arr.reshape(n_nodes, ndof_per_node)
    if arr.ndim == 2:
        if arr.shape != (n_nodes, ndof_per_node):
            raise ValueError(
                f"{name} shape mismatch: got {arr.shape}, expected {(n_nodes, ndof_per_node)}."
            )
        return arr
    raise ValueError(f"Unsupported nodal field shape for {name}: {arr.shape}")


def _build_mat_mesh_view_from_raw(raw: dict) -> _MatMeshView:
    if "InpData" not in raw:
        raise ValueError("JAX MAT is missing InpData; cannot build MAT-native validation geometry.")

    inp_struct = _unwrap_scalar(raw["InpData"])
    points_raw = inp_struct.get("points") if isinstance(inp_struct, dict) else getattr(inp_struct, "points", None)
    if points_raw is None:
        raise ValueError("JAX MAT InpData is missing points.")
    points = np.asarray(points_raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError(f"Unsupported InpData.points shape: {points.shape}")

    type_names_raw = inp_struct.get("cell_type_names") if isinstance(inp_struct, dict) else getattr(inp_struct, "cell_type_names", None)
    cell_sets_raw = inp_struct.get("cell_type_cells") if isinstance(inp_struct, dict) else getattr(inp_struct, "cell_type_cells", None)
    if type_names_raw is None or cell_sets_raw is None:
        raise ValueError("JAX MAT InpData is missing cell_type_names/cell_type_cells.")

    type_names = np.asarray(type_names_raw, dtype=object).reshape(-1)
    cell_sets = np.asarray(cell_sets_raw, dtype=object).reshape(-1)
    if type_names.size != cell_sets.size or type_names.size == 0:
        raise ValueError(
            "InpData cell_type_names/cell_type_cells size mismatch: "
            f"{type_names.size} vs {cell_sets.size}."
        )

    blocks: list[_MatMeshBlock] = []
    for raw_name, raw_cells in zip(type_names.tolist(), cell_sets.tolist()):
        cell_type_name = _as_optional_str(raw_name, "")
        if not cell_type_name:
            raise ValueError("Encountered empty cell type name inside InpData.")
        ele_type = DEFAULT_ELE_TYPE_FROM_CELL_TYPE.get(cell_type_name.lower(), cell_type_name.upper())
        connectivity = np.asarray(_unwrap_scalar(raw_cells), dtype=np.int32)
        if connectivity.ndim == 1:
            connectivity = connectivity.reshape(1, -1)
        if connectivity.ndim != 2:
            raise ValueError(
                f"Unsupported connectivity shape for cell type '{cell_type_name}': {connectivity.shape}"
            )
        if connectivity.size == 0:
            continue
        blocks.append(_MatMeshBlock(ele_type=str(ele_type).upper(), connectivity=connectivity))

    if not blocks:
        raise ValueError("InpData does not contain any non-empty cell blocks.")

    return _MatMeshView(points=points, blocks=tuple(blocks))


def _block_result_view_from_mat(
    mesh_view: _MatMeshView,
    result: AnalysisResult,
    elem_offset: int,
    n_elem: int,
    field_name: str,
) -> np.ndarray:
    field = getattr(result, field_name)
    if field is None:
        raise ValueError(f"AnalysisResult.{field_name} is required for solid ODB validation.")

    arr = np.asarray(field, dtype=np.float64)
    layout = str(result.metadata.get("gauss_layout", "")).strip().lower()
    if layout == "element_gauss" or (not layout and arr.ndim >= 1 and arr.shape[0] == mesh_view.n_elements):
        return arr[elem_offset: elem_offset + n_elem]

    if result.n_gauss_per_elem is None:
        raise ValueError("AnalysisResult.n_gauss_per_elem is required for flat Gauss layouts.")

    gauss_counts = np.asarray(result.n_gauss_per_elem, dtype=np.int32).reshape(-1)
    block_counts = gauss_counts[elem_offset: elem_offset + n_elem]
    unique_counts = np.unique(block_counts)
    if unique_counts.size != 1:
        raise ValueError(
            "Mixed Gauss counts inside one block are not supported: "
            f"{unique_counts.tolist()}"
        )

    n_gp = int(unique_counts[0])
    gp_offset = int(np.sum(gauss_counts[:elem_offset]))
    block_flat = arr[gp_offset: gp_offset + n_elem * n_gp]
    return block_flat.reshape((n_elem, n_gp) + arr.shape[1:])


def _block_result_view(
    result: AnalysisResult,
    model: Model,
    block_index: int,
    field_name: str,
) -> np.ndarray:
    field = getattr(result, field_name)
    if field is None:
        raise ValueError(f"AnalysisResult.{field_name} is required for solid ODB validation.")

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


def _analysis_nodal_data(model: Model, result: AnalysisResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ndof_per_node = _analysis_ndof_per_node(model, result)
    n_nodes = model.mesh.n_nodes
    coords = np.asarray(model.mesh.nodes, dtype=np.float64)
    disp = _reshape_nodal_field(result.u, n_nodes, ndof_per_node, "u")
    # Historical validation MATs store resisting nodal forces with the ABAQUS sign.
    nforc = -_reshape_nodal_field(result.internal_force, n_nodes, ndof_per_node, "internal_force")
    return coords, disp, nforc


def _analysis_nodal_data_from_mat(mesh_view: _MatMeshView, result: AnalysisResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ndof_per_node = _analysis_ndof_per_node_from_mat(mesh_view, result)
    coords = np.asarray(mesh_view.points, dtype=np.float64)
    disp = _reshape_nodal_field(result.u, mesh_view.n_nodes, ndof_per_node, "u")
    nforc = -_reshape_nodal_field(result.internal_force, mesh_view.n_nodes, ndof_per_node, "internal_force")
    return coords, disp, nforc


def _analysis_element_blocks(model: Model, result: AnalysisResult) -> dict[str, dict[str, np.ndarray]]:
    blocks: dict[str, dict[str, list[np.ndarray]]] = {}
    for block_index, block in enumerate(model.mesh.blocks):
        ele_type = str(block.ele_type).upper()
        if ele_type not in SUPPORTED_SOLID_ELE_TYPES:
            continue
        coords = np.asarray(model.mesh.nodes[np.asarray(block.connectivity, dtype=np.int32)], dtype=np.float64)
        stress = _reorder_block_gauss_to_legacy(
            ele_type,
            _block_result_view(result, model, block_index, "gauss_stress"),
        )
        strain = _reorder_block_gauss_to_legacy(
            ele_type,
            _block_result_view(result, model, block_index, "gauss_strain"),
        )
        entry = blocks.setdefault(ele_type, {"coords": [], "stress": [], "strain": []})
        entry["coords"].append(coords)
        entry["stress"].append(np.asarray(stress, dtype=np.float64))
        entry["strain"].append(np.asarray(strain, dtype=np.float64))

    return {
        ele_type: {
            "coords": np.concatenate(data["coords"], axis=0),
            "stress": np.concatenate(data["stress"], axis=0),
            "strain": np.concatenate(data["strain"], axis=0),
        }
        for ele_type, data in blocks.items()
    }


def _analysis_element_blocks_from_mat(
    mesh_view: _MatMeshView,
    result: AnalysisResult,
) -> dict[str, dict[str, np.ndarray]]:
    blocks: dict[str, dict[str, list[np.ndarray]]] = {}
    elem_offset = 0
    for block in mesh_view.blocks:
        n_elem = int(block.connectivity.shape[0])
        ele_type = str(block.ele_type).upper()
        connectivity = np.asarray(block.connectivity, dtype=np.int32)
        if ele_type not in SUPPORTED_SOLID_ELE_TYPES:
            elem_offset += n_elem
            continue
        coords = np.asarray(mesh_view.points[connectivity], dtype=np.float64)
        stress = _reorder_block_gauss_to_legacy(
            ele_type,
            _block_result_view_from_mat(mesh_view, result, elem_offset, n_elem, "gauss_stress"),
        )
        strain = _reorder_block_gauss_to_legacy(
            ele_type,
            _block_result_view_from_mat(mesh_view, result, elem_offset, n_elem, "gauss_strain"),
        )
        entry = blocks.setdefault(ele_type, {"coords": [], "stress": [], "strain": []})
        entry["coords"].append(coords)
        entry["stress"].append(np.asarray(stress, dtype=np.float64))
        entry["strain"].append(np.asarray(strain, dtype=np.float64))
        elem_offset += n_elem

    return {
        ele_type: {
            "coords": np.concatenate(data["coords"], axis=0),
            "stress": np.concatenate(data["stress"], axis=0),
            "strain": np.concatenate(data["strain"], axis=0),
        }
        for ele_type, data in blocks.items()
    }


def _analysis_scalar_blocks_from_mat(
    mesh_view: _MatMeshView,
    result: AnalysisResult,
    field_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    blocks: dict[str, dict[str, list[np.ndarray]]] = {}
    elem_offset = 0
    for block in mesh_view.blocks:
        n_elem = int(block.connectivity.shape[0])
        ele_type = str(block.ele_type).upper()
        connectivity = np.asarray(block.connectivity, dtype=np.int32)
        if ele_type not in SUPPORTED_SOLID_ELE_TYPES:
            elem_offset += n_elem
            continue
        coords = np.asarray(mesh_view.points[connectivity], dtype=np.float64)
        values = _block_result_view_from_mat(mesh_view, result, elem_offset, n_elem, field_name)
        entry = blocks.setdefault(ele_type, {"coords": [], "value": []})
        entry["coords"].append(coords)
        entry["value"].append(np.asarray(values, dtype=np.float64))
        elem_offset += n_elem

    return {
        ele_type: {
            "coords": np.concatenate(data["coords"], axis=0),
            "value": np.concatenate(data["value"], axis=0),
        }
        for ele_type, data in blocks.items()
    }


def _normalize_nforc_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[-1]
    return arr


def _mat_has_legacy_solid_fields(raw: dict) -> bool:
    return (
        "InpData" in raw
        and ("EleGaussStressSet_jax" in raw or "EleGaussStressSetByType_jax" in raw)
        and ("EleGaussStrainSet_jax" in raw or "EleGaussStrainSetByType_jax" in raw)
        and ("NodalDisp_jax" in raw or "EleNodeDispSet_jax" in raw)
    )


def _mat_has_unified_result_fields(raw: dict) -> bool:
    has_disp = "u" in raw or "solid_u_nodal" in raw
    return has_disp and all(key in raw for key in ("gauss_stress", "gauss_strain"))


def _mat_has_frame_result_fields(raw: dict) -> bool:
    required = {"frame_u", "frame_internal_force", "frame_gauss_stress", "frame_gauss_strain"}
    return "InpData" in raw and required.issubset(raw.keys())


def _frame_index_from_raw(raw: dict, frame_index: int | None = None) -> int:
    frame_count = int(np.asarray(raw["frame_u"]).shape[0])
    if frame_count <= 0:
        raise ValueError("JAX frame MAT does not contain any frames.")
    if frame_index is None:
        return frame_count - 1
    idx = int(frame_index)
    if idx < 0:
        idx = frame_count + idx
    if idx < 0 or idx >= frame_count:
        raise IndexError(f"frame_index {frame_index} out of range for {frame_count} frames.")
    return idx


def _analysis_result_from_frame_mat(raw: dict, frame_index: int | None = None) -> AnalysisResult:
    """Build a single-frame AnalysisResult snapshot from nonlinear history MAT fields."""
    idx = _frame_index_from_raw(raw, frame_index)
    metadata: dict[str, object] = {}
    try:
        if "metadata" in raw:
            metadata.update(_mat_struct_to_dict(raw["metadata"]))
    except Exception:
        pass
    metadata["analysis_type"] = str(metadata.get("analysis_type") or "nonlinear_static")
    metadata["frame_index"] = idx
    metadata["gauss_layout"] = str(metadata.get("gauss_layout") or "element_gauss")
    if "frame_increment_index" in raw:
        metadata["increment_index"] = int(np.asarray(raw["frame_increment_index"], dtype=np.int32).reshape(-1)[idx])
    if "frame_load_scale" in raw:
        metadata["load_scale"] = float(np.asarray(raw["frame_load_scale"], dtype=np.float64).reshape(-1)[idx])
    if "frame_converged" in raw:
        metadata["converged"] = bool(np.asarray(raw["frame_converged"], dtype=np.int32).reshape(-1)[idx])
    if "frame_n_iterations" in raw:
        metadata["n_iterations"] = int(np.asarray(raw["frame_n_iterations"], dtype=np.int32).reshape(-1)[idx])

    gauss_peeq = None
    if "frame_gauss_peeq" in raw:
        gauss_peeq = np.asarray(raw["frame_gauss_peeq"], dtype=np.float64)[idx]

    return AnalysisResult(
        u=np.asarray(raw["frame_u"], dtype=np.float64)[idx],
        reaction=np.asarray(raw["frame_reaction"], dtype=np.float64)[idx] if "frame_reaction" in raw else None,
        gauss_stress=np.asarray(raw["frame_gauss_stress"], dtype=np.float64)[idx],
        gauss_strain=np.asarray(raw["frame_gauss_strain"], dtype=np.float64)[idx],
        gauss_peeq=gauss_peeq,
        internal_force=np.asarray(raw["frame_internal_force"], dtype=np.float64)[idx],
        gauss_coords=np.asarray(raw["gauss_coords"], dtype=np.float64) if "gauss_coords" in raw else None,
        gauss_vols=np.asarray(raw["gauss_vols"], dtype=np.float64) if "gauss_vols" in raw else None,
        n_gauss_per_elem=np.asarray(raw["n_gauss_per_elem"], dtype=np.int32).reshape(-1) if "n_gauss_per_elem" in raw else None,
        metadata=metadata,
    )


def _inp_path_from_result_mat(raw: dict) -> str | None:
    if "InpData" in raw:
        inp_struct = _unwrap_scalar(raw["InpData"])
        if isinstance(inp_struct, dict):
            for key in ("inp_path", "source_file"):
                if key in inp_struct:
                    text = _as_optional_str(inp_struct[key], "")
                    if text:
                        return text
            return None
        return _as_optional_str(getattr(inp_struct, "inp_path", None), "") or _as_optional_str(
            getattr(inp_struct, "source_file", None),
            "",
        ) or None

    if "metadata" not in raw:
        return None

    meta_raw = raw["metadata"]
    metadata: dict[str, object] = {}
    try:
        metadata = _mat_struct_to_dict(meta_raw)
    except TypeError:
        if isinstance(meta_raw, dict):
            metadata = meta_raw
        elif isinstance(meta_raw, np.ndarray) and getattr(meta_raw.dtype, "names", None):
            rec = meta_raw.reshape(-1)[0]
            metadata = {name: rec[name] for name in meta_raw.dtype.names}

    for key in ("source_file", "inp_path"):
        if key in metadata:
            text = _as_optional_str(metadata[key], "")
            if text:
                return text
    return None


def compute_validation_errors_from_mat(
    jax_mat_path: str | Path,
    abaqus_mat_path: str | Path,
) -> dict[str, float]:
    """Compare one saved JAX MAT payload against one ABAQUS MAT export."""
    distributions = compute_validation_error_distributions_from_mat(jax_mat_path, abaqus_mat_path)
    return {
        key: float(np.max(values)) if np.asarray(values).size else 0.0
        for key, values in distributions.items()
    }


def compute_validation_error_distributions_from_mat(
    jax_mat_path: str | Path,
    abaqus_mat_path: str | Path,
) -> dict[str, np.ndarray]:
    """Return row-wise L1 error distributions for one JAX-vs-ABAQUS MAT pair."""
    jax_raw = _load_mat(jax_mat_path)
    aba_raw = _load_mat(abaqus_mat_path)

    if _mat_has_legacy_solid_fields(jax_raw):
        jax_coord_dim = int(np.asarray(_unwrap_scalar(jax_raw["InpData"]).points, dtype=np.float64).shape[1])
        jax_vec_dim, _ = _infer_jax_vec_dim(jax_raw)

        jax_coords, jax_u_dense = _jax_nodal_data(jax_raw)
        if "NFORC_jax" not in jax_raw:
            raise ValueError(f"Legacy MAT does not contain NFORC_jax: {jax_mat_path}")
        jax_nforc_dense = _normalize_nforc_array(jax_raw["NFORC_jax"])
        jax_blocks = _jax_element_blocks(jax_raw)
    elif _mat_has_unified_result_fields(jax_raw):
        result = AnalysisResult.from_mat(str(jax_mat_path))
        mesh_view = _build_mat_mesh_view_from_raw(jax_raw)
        jax_coord_dim = mesh_view.coord_dim
        jax_vec_dim = _analysis_ndof_per_node_from_mat(mesh_view, result)
        jax_coords, jax_u_dense, jax_nforc_dense = _analysis_nodal_data_from_mat(mesh_view, result)
        jax_blocks = _analysis_element_blocks_from_mat(mesh_view, result)
    elif _mat_has_frame_result_fields(jax_raw):
        result = _analysis_result_from_frame_mat(jax_raw)
        mesh_view = _build_mat_mesh_view_from_raw(jax_raw)
        jax_coord_dim = mesh_view.coord_dim
        jax_vec_dim = _analysis_ndof_per_node_from_mat(mesh_view, result)
        jax_coords, jax_u_dense, jax_nforc_dense = _analysis_nodal_data_from_mat(mesh_view, result)
        jax_blocks = _analysis_element_blocks_from_mat(mesh_view, result)
    else:
        raise ValueError(
            "Unsupported JAX MAT payload for ODB validation. "
            "Expected legacy *_jax fields, unified AnalysisResult fields, or nonlinear frame_* fields."
        )

    aba_labels, aba_coords = _abaqus_node_data_from_legacy_fields(aba_raw)
    del aba_labels
    aba_u = _legacy_nodal_disp(aba_raw["EleNodeLabelSet"], aba_raw["EleNodeDispSet"])
    aba_nforc = np.asarray(aba_raw["NFORC"], dtype=np.float64)[-1]
    aba_coords = _match_trailing_dim(aba_coords, jax_coord_dim)
    aba_u = _match_trailing_dim(aba_u, jax_vec_dim)
    aba_nforc = _match_trailing_dim(aba_nforc, jax_vec_dim)

    jax_coords = _match_trailing_dim(jax_coords, jax_coord_dim)
    jax_u_dense = _match_trailing_dim(jax_u_dense, jax_vec_dim)
    jax_nforc_dense = _match_trailing_dim(jax_nforc_dense, jax_vec_dim)
    jax_u = _reorder_jax_nodal_to_abaqus(aba_coords, jax_coords, jax_u_dense)
    jax_nforc = _reorder_jax_nodal_to_abaqus(aba_coords, jax_coords, jax_nforc_dense)

    aba_blocks = _abaqus_element_blocks(aba_raw)
    aba_s_parts = []
    aba_e_parts = []
    jax_s_parts = []
    jax_e_parts = []
    for ele_type in sorted(aba_blocks):
        if ele_type not in jax_blocks:
            raise KeyError(f"JAX MAT is missing element block for Abaqus element type '{ele_type}'.")
        aba_block = aba_blocks[ele_type]
        jax_s_aligned, jax_e_aligned = _align_jax_block_to_abaqus(aba_block["coords"], jax_blocks[ele_type])
        aba_s_parts.append(np.asarray(aba_block["stress"], dtype=np.float64))
        aba_e_parts.append(np.asarray(aba_block["strain"], dtype=np.float64))
        jax_s_parts.append(jax_s_aligned)
        jax_e_parts.append(jax_e_aligned)

    aba_s = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in aba_s_parts], axis=0)
    aba_e = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in aba_e_parts], axis=0)
    jax_s = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in jax_s_parts], axis=0)
    jax_e = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in jax_e_parts], axis=0)

    return {
        "U": _norm1_error_distribution(aba_u, jax_u),
        "S": _norm1_error_distribution(aba_s, jax_s),
        "E": _norm1_error_distribution(aba_e, jax_e),
        "NFORC": _norm1_error_distribution(aba_nforc, jax_nforc),
    }


def _max_norm1_error(aba: np.ndarray, jax: np.ndarray) -> float:
    return float(np.max(_norm1_error_distribution(aba, jax)))


def _norm1_error_distribution(aba: np.ndarray, jax: np.ndarray) -> np.ndarray:
    aba_arr = np.asarray(aba, dtype=np.float64)
    jax_arr = np.asarray(jax, dtype=np.float64)
    if aba_arr.shape != jax_arr.shape:
        raise ValueError(f"Shape mismatch: ABAQUS {aba_arr.shape} vs JAX {jax_arr.shape}")
    flat_diff = np.abs(jax_arr - aba_arr).reshape(-1, aba_arr.shape[-1])
    return np.sum(flat_diff, axis=1)


def compute_validation_errors(
    model: Model,
    result: AnalysisResult,
    abaqus_mat_path: str | Path,
) -> dict[str, float]:
    aba_raw = _load_mat(abaqus_mat_path)
    jax_vec_dim = _analysis_ndof_per_node(model, result)
    jax_coord_dim = model.mesh.n_dim

    aba_labels, aba_coords = _abaqus_node_data_from_legacy_fields(aba_raw)
    del aba_labels
    aba_u = _legacy_nodal_disp(aba_raw["EleNodeLabelSet"], aba_raw["EleNodeDispSet"])
    aba_nforc = np.asarray(aba_raw["NFORC"], dtype=np.float64)[-1]
    aba_coords = _match_trailing_dim(aba_coords, jax_coord_dim)
    aba_u = _match_trailing_dim(aba_u, jax_vec_dim)
    aba_nforc = _match_trailing_dim(aba_nforc, jax_vec_dim)

    jax_coords, jax_u_dense, jax_nforc_dense = _analysis_nodal_data(model, result)
    jax_coords = _match_trailing_dim(jax_coords, jax_coord_dim)
    jax_u_dense = _match_trailing_dim(jax_u_dense, jax_vec_dim)
    jax_u = _reorder_jax_nodal_to_abaqus(aba_coords, jax_coords, jax_u_dense)
    jax_nforc_dense = _match_trailing_dim(jax_nforc_dense, jax_vec_dim)
    jax_nforc = _reorder_jax_nodal_to_abaqus(aba_coords, jax_coords, jax_nforc_dense)

    aba_blocks = _abaqus_element_blocks(aba_raw)
    jax_blocks = _analysis_element_blocks(model, result)
    aba_s_parts = []
    aba_e_parts = []
    jax_s_parts = []
    jax_e_parts = []
    for ele_type in sorted(aba_blocks):
        if ele_type not in jax_blocks:
            raise KeyError(f"JAX result is missing element block for Abaqus element type '{ele_type}'.")
        aba_block = aba_blocks[ele_type]
        jax_s_aligned, jax_e_aligned = _align_jax_block_to_abaqus(aba_block["coords"], jax_blocks[ele_type])
        aba_s_parts.append(np.asarray(aba_block["stress"], dtype=np.float64))
        aba_e_parts.append(np.asarray(aba_block["strain"], dtype=np.float64))
        jax_s_parts.append(jax_s_aligned)
        jax_e_parts.append(jax_e_aligned)

    aba_s = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in aba_s_parts], axis=0)
    aba_e = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in aba_e_parts], axis=0)
    jax_s = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in jax_s_parts], axis=0)
    jax_e = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1]) for arr in jax_e_parts], axis=0)

    return {
        "U": _max_norm1_error(aba_u, jax_u),
        "S": _max_norm1_error(aba_s, jax_s),
        "E": _max_norm1_error(aba_e, jax_e),
        "NFORC": _max_norm1_error(aba_nforc, jax_nforc),
    }


def _build_error_struct(errors: dict[str, float]):
    payload = {
        "metric": np.asarray(["max_norm1"]),
    }
    preferred_keys = [
        "U",
        "S",
        "E",
        "NFORC",
        "PEEQ",
        "PEEQ_time_max",
        "PEEQ_time_mean",
        "PEEQ_final",
    ]
    ordered_keys = preferred_keys + [key for key in sorted(errors) if key not in preferred_keys]
    for key in ordered_keys:
        if key not in errors:
            continue
        arr = np.asarray(errors[key], dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        payload[key] = arr.reshape(-1, 1)
    return payload


def _abaqus_scalar_field_from_legacy_set(aba_raw: dict, key: str) -> np.ndarray:
    if key not in aba_raw:
        return np.empty((0,), dtype=np.float64)
    field_raw = np.asarray(aba_raw[key], dtype=object)
    if field_raw.ndim >= 2 and field_raw.shape[0] > 1:
        field_objs = [field_raw[i] for i in range(field_raw.shape[0])]
    else:
        field_objs = field_raw.reshape(-1)
    values: list[np.ndarray] = []
    for obj in field_objs:
        arr = np.asarray(obj, dtype=object)
        flat = [float(np.asarray(_unwrap_scalar(x), dtype=np.float64).reshape(-1)[0]) for x in arr.reshape(-1)]
        values.append(np.asarray(flat, dtype=np.float64))
    return np.concatenate(values, axis=0) if values else np.empty((0,), dtype=np.float64)


def _obsolete_abaqus_payload_keys() -> list[str]:
    return [
        "ErrorHistory_with_ODB",
        "info",
        "odb_path",
        "frame_time",
        "voigt_order",
        "ip_order",
        "jax_from_aba",
        "voigt_perm",
        "NFORC",
        "EleLabelSet",
        "EleTypeSet",
        "EleMaterialIDSet",
        "NodeConstrainedDof1",
        "NodeConstrainedDof2",
        "NodeConstrainedDof3",
        "EleNodeLabelSet",
        "EleNodeCoordSet",
        "EleNodeDispSet",
        "EleGaussStressSet",
        "EleGaussStrainSet",
        "EleGaussPEEQSet",
        "instance_name",
        "step_name",
        "frame_index",
        "frame_value",
        "U",
        "S",
        "E",
        "NodeLabels",
        "NodeCoords",
        "ElemLabels",
        "ElemConn",
        "ElemTypes",
        "abaqus_node_labels",
        "abaqus_node_coords",
        "abaqus_u_nodal",
        "abaqus_nforc",
        "abaqus_gauss_stress",
        "abaqus_gauss_strain",
        "abaqus_gauss_peeq",
        "abaqus_elem_types",
        "abaqus_frame_time",
        "abaqus_peeq_time_max",
        "abaqus_peeq_time_mean",
        "abaqus_voigt_order",
        "abaqus_ip_order",
        "abaqus_jax_from_aba",
        "abaqus_voigt_perm",
        "info_abaqus",
        "abaqus_odb_path",
        "abaqus_instance_name",
        "abaqus_step_name",
        "abaqus_frame_index",
        "abaqus_frame_value",
    ]


def _abaqus_solid_payload_from_raw(aba_raw: dict) -> dict:
    aba_labels, aba_coords = _abaqus_node_data_from_legacy_fields(aba_raw)
    aba_elem_labels, aba_elem_conn, aba_elem_types = _abaqus_element_topology_from_raw(aba_raw)
    aba_u = _legacy_nodal_disp(aba_raw["EleNodeLabelSet"], aba_raw["EleNodeDispSet"])
    aba_nforc = np.asarray(aba_raw["NFORC"], dtype=np.float64)[-1]
    aba_blocks = _abaqus_element_blocks(aba_raw)
    aba_peeq = _abaqus_scalar_field_from_legacy_set(aba_raw, "EleGaussPEEQSet")
    aba_s = np.concatenate(
        [
            np.concatenate(
                [np.asarray(arr, dtype=np.float64).reshape(-1, np.asarray(arr, dtype=np.float64).shape[-1])
                 for arr in aba_blocks[ele]["stress"]],
                axis=0,
            )
            for ele in sorted(aba_blocks)
        ],
        axis=0,
    )
    aba_e = np.concatenate(
        [
            np.concatenate(
                [np.asarray(arr, dtype=np.float64).reshape(-1, np.asarray(arr, dtype=np.float64).shape[-1])
                 for arr in aba_blocks[ele]["strain"]],
                axis=0,
            )
            for ele in sorted(aba_blocks)
        ],
        axis=0,
    )

    payload = {
        "abaqus_node_labels": np.asarray(aba_labels, dtype=np.int32).reshape(-1, 1),
        "abaqus_node_coords": np.asarray(aba_coords, dtype=np.float64),
        "abaqus_elem_labels": np.asarray(aba_elem_labels, dtype=np.int32).reshape(-1, 1),
        "abaqus_elem_conn": aba_elem_conn,
        "abaqus_elem_types": aba_elem_types,
        "abaqus_u_nodal": np.asarray(aba_u, dtype=np.float64),
        "abaqus_nforc": np.asarray(aba_nforc, dtype=np.float64),
        "abaqus_gauss_stress": np.asarray(aba_s, dtype=np.float64),
        "abaqus_gauss_strain": np.asarray(aba_e, dtype=np.float64),
        "abaqus_gauss_peeq": np.asarray(aba_peeq, dtype=np.float64),
    }

    if "frame_time" in aba_raw:
        payload["abaqus_frame_time"] = np.asarray(aba_raw["frame_time"], dtype=np.float64).reshape(-1)
    if "PEEQ_time_max" in aba_raw:
        payload["abaqus_peeq_time_max"] = np.asarray(aba_raw["PEEQ_time_max"], dtype=np.float64).reshape(-1)
    if "PEEQ_time_mean" in aba_raw:
        payload["abaqus_peeq_time_mean"] = np.asarray(aba_raw["PEEQ_time_mean"], dtype=np.float64).reshape(-1)

    optional_map = {
        "info_abaqus": "info",
        "abaqus_odb_path": "odb_path",
    }
    for out_key, src_key in optional_map.items():
        if src_key in aba_raw:
            payload[out_key] = aba_raw[src_key]
    return payload


def validate_model_result_with_abaqus_odb(
    model: Model,
    result: AnalysisResult,
    abaqus_mat_path: str | Path,
    *,
    validation_payload_path: str | Path | None = None,
    validated_odb_name: str = "(pre-extracted)",
) -> dict:
    """Validate one native solid analysis result against one ABAQUS MAT export."""
    aba_raw = _load_mat(abaqus_mat_path)
    errors = compute_validation_errors(model, result, abaqus_mat_path)

    if validation_payload_path is not None:
        payload_path = Path(validation_payload_path)
        if payload_path.is_file():
            payload = _clean_payload(_load_mat(payload_path))
        else:
            payload = {}
        for key in _obsolete_abaqus_payload_keys():
            payload.pop(key, None)
        payload.update(
            {
                **_abaqus_solid_payload_from_raw(aba_raw),
                "validated_with_ODB": np.asarray([[1]], dtype=np.int32),
                "validated_ODBName": np.asarray([validated_odb_name]),
                "validated_at_ODB": np.asarray([dt.datetime.now().isoformat()]),
                "Error_with_ODB": _build_error_struct(errors),
            }
        )
        _save_mat(payload_path, payload)

    return {
        "performed": True,
        "skipped": False,
        "validated_with_ODB": True,
        "validated_ODBName": validated_odb_name,
        "abaqus_mat_path": str(abaqus_mat_path),
        "errors": errors,
    }


def validate_mat_with_abaqus_odb(
    mat_path: str | Path,
    force: bool = False,
    temp_out_dir: str | Path | None = None,
    config: dict | None = None,
    abaqus_mat_path: str | Path | None = None,
    validated_odb_name: str | None = None,
) -> dict:
    """Validate a JAX MAT against ABAQUS ODB.

    Parameters
    ----------
    abaqus_mat_path : str | Path | None
        If given, skip ODB extraction and use this pre-extracted ABAQUS MAT
        for comparison.  This is the recommended approach under MockWSL
        where WSL→Windows ``cmd.exe`` interop is unreliable.
    """
    mat_path = Path(mat_path)
    raw = _load_mat(mat_path)
    payload = _clean_payload(raw)

    if not force and "validated_with_ODB" in payload and _as_int(payload["validated_with_ODB"]) == 1:
        return {
            "performed": False,
            "skipped": True,
            "validated_with_ODB": True,
            "errors": _mat_struct_to_dict(raw["Error_with_ODB"]) if "Error_with_ODB" in raw else None,
        }

    inp_path = _inp_path_from_result_mat(raw)
    if not inp_path and config is None:
        raise ValueError(
            f"MAT file does not expose an inp_path/source_file for ODB inference: {mat_path}"
        )

    odb_path = validated_odb_name or "(pre-extracted)"
    if abaqus_mat_path is not None:
        abaqus_mat_path = str(abaqus_mat_path)
    else:
        if config is None:
            if "InpData" not in raw:
                raise ValueError(
                    "Automatic ODB inference without config currently requires InpData in the MAT payload."
                )
            odb_path = _infer_odb_path_from_mat(raw)
            odb_name = os.path.basename(odb_path)
            windows_dir = os.path.dirname(odb_path)
        else:
            case_name = mat_path.stem
            candidate_inp_path = inp_path
            if not candidate_inp_path:
                resolved = resolve_model_case_paths(config, case_name)
                candidate_inp_path = str(resolved["inp_path"])
            windows_dir, odb_name = _ensure_windows_odb_from_inp(config, case_name, candidate_inp_path)
            odb_path = str(Path(windows_dir) / odb_name)
        if temp_out_dir is None:
            temp_out_dir = mat_path.parent / "TempValidateODB"
        temp_out_dir = Path(temp_out_dir)
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        abaqus_mat_path = extract_solid_odb(
            win_job_dir=windows_dir,
            odb_name=odb_name,
            abaqus_cmd=str(config.get("windows_abaqus_cmd", "abaqus")).strip() if config is not None else "abaqus",
            linux_out_dir=str(temp_out_dir),
            skip_initial=False,
            step=None,
            fixed_dof1_set=None,
            fixed_dof2_set=None,
            fixed_dof3_set=None,
        )

    aba_raw = _load_mat(abaqus_mat_path)
    errors = compute_validation_errors_from_mat(mat_path, abaqus_mat_path)
    for key in _obsolete_abaqus_payload_keys():
        payload.pop(key, None)
    payload.update(
        {
            **_abaqus_solid_payload_from_raw(aba_raw),
            "validated_with_ODB": np.asarray([[1]], dtype=np.int32),
            "validated_ODBName": np.asarray([odb_path]),
            "validated_at_ODB": np.asarray([dt.datetime.now().isoformat()]),
            "Error_with_ODB": _build_error_struct(errors),
        }
    )
    _save_mat(mat_path, payload)
    return {
        "performed": True,
        "skipped": False,
        "validated_with_ODB": True,
        "validated_ODBName": odb_path,
        "abaqus_mat_path": str(abaqus_mat_path),
        "errors": errors,
    }


def _find_abaqus_mat_for_case(
    config: dict, case_name: str, abaqus_mat_dir: str | Path | None = None,
) -> Path | None:
    """Search for a pre-extracted ABAQUS MAT file for the given case.

    Search order:
    1. ``abaqus_mat_dir/<case_name>.mat``
    2. ``<win_stored_root>/<model_name>/<case_name>.mat``
    3. ``<win_stored_root>/<model_name>/abaqus/<case_name>.mat``
    """
    candidates = []
    if abaqus_mat_dir is not None:
        candidates.append(Path(abaqus_mat_dir) / f"{case_name}.mat")

    win_root_wsl = win_to_wsl_path(config["win_stored_root"])
    model_name = str(config["model_name"])
    candidates.extend([
        win_root_wsl / model_name / f"{case_name}.mat",
        win_root_wsl / model_name / "abaqus" / f"{case_name}.mat",
    ])
    for c in candidates:
        if c.is_file():
            return c
    return None


def _print_extraction_commands(config: dict) -> None:
    """Print Windows PowerShell commands to extract ODB data for all load cases."""
    win_root_wsl = win_to_wsl_path(config["win_stored_root"])
    model_name = str(config["model_name"])
    abaqus_cmd = str(config.get("windows_abaqus_cmd", "abaqus")).strip() or "abaqus"
    exporter_src = os.path.join(PROJECT_ROOT, "jaxmech", "io", "abaqus", "export_unified_odb.py")
    exporter_win = _to_windows_path_from_wsl(exporter_src)

    for case_name in config["load_cases"]:
        # Search for ODB
        odb_candidates = [
            win_root_wsl / model_name / f"{case_name}.odb",
            win_root_wsl / model_name / "abaqus" / f"{case_name}.odb",
            win_root_wsl / model_name / case_name / f"{case_name}.odb",
        ]
        odb_wsl = None
        for c in odb_candidates:
            if c.is_file():
                odb_wsl = c
                break
        if odb_wsl is None:
            print(f"  WARNING: ODB not found for {case_name}, searched:")
            for c in odb_candidates:
                print(f"    {c}")
            continue

        odb_dir_win = _to_windows_path_from_wsl(str(odb_wsl.parent))
        odb_win = _to_windows_path_from_wsl(str(odb_wsl))
        prefix_win = _to_windows_path_from_wsl(str(odb_wsl.with_suffix("")))
        print(f"  # {case_name}:")
        print(f"  cmd /c \"cd /d {odb_dir_win} && {abaqus_cmd} python {exporter_win} --odb {odb_win} --out-prefix {prefix_win}\"")


def run_odb_validation(
    config_path: str | Path,
    config: dict | None = None,
    force: bool = False,
    abaqus_mat_dir: str | Path | None = None,
    skip_extraction: bool = False,
) -> list[dict]:
    """Validate all configured solid load cases against ABAQUS ODB.

    Parameters
    ----------
    abaqus_mat_dir : str | Path | None
        Directory containing pre-extracted ABAQUS MAT files.  When given,
        ``run_extraction()`` is skipped and the comparison is done directly.
        This is the recommended approach under MockWSL.
    skip_extraction : bool
        If True, skip ``run_extraction()`` and search for pre-extracted
        ABAQUS MATs automatically (in ``win_stored_root`` locations).
        This is equivalent to ``--step compare`` in CLI.
    """
    config_path = str(config_path)
    config = parse_shakedown_config(config_path) if config is None else dict(config)
    jax_system = normalize_jax_system(config["jax_system"])
    if jax_system == "WIN":
        raise NotImplementedError(
            "jax_system=WIN is defined in config, but the current run_odb_validation workflow "
            "is only implemented from the WSL side."
        )

    results = []
    for case_name in config["load_cases"]:
        resolved = resolve_model_case_paths(config, case_name)
        if not resolved["mat_path"].is_file():
            raise FileNotFoundError(
                f"JAX MAT not found for {case_name}: {resolved['mat_path']}. "
                f"Run the elastic build step first so validation starts from MAT."
            )

        aba_mat = None
        if skip_extraction or abaqus_mat_dir is not None:
            aba_mat = _find_abaqus_mat_for_case(config, case_name, abaqus_mat_dir)
            if aba_mat is None:
                raise FileNotFoundError(
                    f"Pre-extracted ABAQUS MAT not found for {case_name}. "
                    f"Run --step extract first on Windows."
                )
            validated_odb_name = "(pre-extracted)"
        else:
            if resolved["odb_path"].is_file():
                windows_dir = str(resolved["odb_path"].parent)
                odb_name = resolved["odb_path"].name
            else:
                windows_dir, odb_name = _ensure_windows_odb_from_inp(config, case_name, resolved["inp_path"])
            validated_odb_name = str(Path(windows_dir) / odb_name)
            temp_out_dir = resolved["jax_dir"] / "TempValidateODB"
            temp_out_dir.mkdir(parents=True, exist_ok=True)
            aba_mat = Path(
                extract_solid_odb(
                    win_job_dir=windows_dir,
                    odb_name=odb_name,
                    abaqus_cmd=str(config.get("windows_abaqus_cmd", "abaqus")).strip() or "abaqus",
                    linux_out_dir=str(temp_out_dir),
                    skip_initial=False,
                    step=None,
                    fixed_dof1_set=None,
                    fixed_dof2_set=None,
                    fixed_dof3_set=None,
                )
            )

        validation = validate_mat_with_abaqus_odb(
            resolved["mat_path"],
            force=force,
            config=config,
            abaqus_mat_path=aba_mat,
            validated_odb_name=validated_odb_name,
        )
        validation["case_name"] = case_name
        results.append(validation)
    return results


def _print_results(results: list[dict]) -> None:
    for result in results:
        if result.get("skipped"):
            print(f"Validation skipped: {result['case_name']} already has validated_with_ODB = 1")
            continue
        print("Validation completed:")
        print(f"  case: {result['case_name']}")
        print(f"  ODB: {result.get('validated_ODBName', 'n/a')}")
        print(f"  ABAQUS MAT: {result.get('abaqus_mat_path', 'n/a')}")
        for key in ["U", "S", "E", "NFORC"]:
            print(f"  max_norm1({key}) = {result['errors'][key]:.6e}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "MockWSL two-step usage:\n"
            "  1) python validate_abaqus_odb.py --config xxx.cfg --step extract\n"
            "     (prints Windows commands to run manually)\n"
            "  2) python validate_abaqus_odb.py --config xxx.cfg --step compare [--force]\n"
            "     (reads pre-extracted ABAQUS MATs and compares)"
        ),
    )
    ap.add_argument("--config", required=True, help="Path to the .cfg file.")
    ap.add_argument("--force", action="store_true", help="Re-run ODB validation even if validated_with_ODB = 1")
    ap.add_argument(
        "--step",
        choices=("all", "extract", "compare"),
        default="all",
        help=(
            "Execution step. 'all' (default) tries cmd.exe interop (pure WSL only). "
            "'extract' prints Windows commands for manual ABAQUS extraction. "
            "'compare' uses pre-extracted ABAQUS MATs from win_stored_root."
        ),
    )
    ap.add_argument(
        "--abaqus-mat-dir",
        default=None,
        help="Directory with pre-extracted ABAQUS MATs (overrides auto-search).",
    )
    args = ap.parse_args()

    config = parse_shakedown_config(args.config)

    if args.step == "extract":
        print("Run these commands in Windows PowerShell to extract ODB data:")
        print()
        _print_extraction_commands(config)
        print()
        print("After extraction, run with --step compare to validate.")
        return

    if args.step == "compare":
        abaqus_mat_dir = args.abaqus_mat_dir
        results = run_odb_validation(
            args.config, config=config, force=args.force,
            abaqus_mat_dir=abaqus_mat_dir,
            skip_extraction=True,
        )
        _print_results(results)
        return

    # --step all: original behavior (cmd.exe interop)
    results = run_odb_validation(args.config, force=args.force)
    _print_results(results)


if __name__ == "__main__":
    main()
