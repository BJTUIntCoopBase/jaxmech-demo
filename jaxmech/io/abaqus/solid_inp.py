"""
jaxmech.io.abaqus.solid_inp
===========================

Native ABAQUS solid INP parser for the current JAX elastic workflow.

Scope
-----
- This parser is intentionally scoped to the current elastic-model path.
- It extracts the information needed so downstream steps can run without
  reopening the source ``.inp`` file.
- More complex future workflows will need richer parsing coverage and a
  merged data model built from both ``.inp`` and companion ``.for`` files.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import re

import numpy as np
import scipy.io as sio

ABAQUS_FACE_INDS_BY_ELE_TYPE = {
    "CPS4": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPS4R": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPS3": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 0], dtype=np.int32),
    },
    "C3D8": {
        "S1": np.array([0, 1, 2, 3], dtype=np.int32),
        "S2": np.array([4, 5, 6, 7], dtype=np.int32),
        "S3": np.array([0, 1, 5, 4], dtype=np.int32),
        "S4": np.array([1, 2, 6, 5], dtype=np.int32),
        "S5": np.array([2, 3, 7, 6], dtype=np.int32),
        "S6": np.array([3, 0, 4, 7], dtype=np.int32),
    },
    "C3D8R": {
        "S1": np.array([0, 1, 2, 3], dtype=np.int32),
        "S2": np.array([4, 5, 6, 7], dtype=np.int32),
        "S3": np.array([0, 1, 5, 4], dtype=np.int32),
        "S4": np.array([1, 2, 6, 5], dtype=np.int32),
        "S5": np.array([2, 3, 7, 6], dtype=np.int32),
        "S6": np.array([3, 0, 4, 7], dtype=np.int32),
    },
    "C3D6": {
        "S1": np.array([0, 1, 2], dtype=np.int32),
        "S2": np.array([3, 4, 5], dtype=np.int32),
        "S3": np.array([0, 1, 4, 3], dtype=np.int32),
        "S4": np.array([1, 2, 5, 4], dtype=np.int32),
        "S5": np.array([2, 0, 3, 5], dtype=np.int32),
    },
    "C3D4": {
        "S1": np.array([0, 1, 2], dtype=np.int32),
        "S2": np.array([0, 3, 1], dtype=np.int32),
        "S3": np.array([1, 3, 2], dtype=np.int32),
        "S4": np.array([2, 3, 0], dtype=np.int32),
    },
}

ABAQUS_TO_MESHIO_CELL_TYPE = {
    "CPS4": "quad",
    "CPS4R": "quad",
    "CPS3": "triangle",
    "C3D8": "hexahedron",
    "C3D8R": "hexahedron",
    "C3D6": "wedge",
    "C3D4": "tetra",
}

DEFAULT_ABAQUS_ELE_TYPE_BY_CELL_TYPE = {
    "quad": "CPS4",
    "quadrilateral": "CPS4",
    "triangle": "CPS3",
    "hexahedron": "C3D8",
    "wedge": "C3D6",
    "tetra": "C3D4",
}


def _read_inp_text(inp_path: str) -> str:
    with open(inp_path, "r", encoding="ascii", errors="ignore") as fh:
        return fh.read()


def parse_materials(inp_text: str) -> dict[str, dict]:
    materials = {}
    current_name = None
    lines = inp_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()
        if upper.startswith("*MATERIAL"):
            match = re.search(r"NAME\s*=\s*([^,\s]+)", line, re.IGNORECASE)
            current_name = match.group(1).upper() if match else None
            if current_name:
                materials[current_name] = {}
            i += 1
            continue
        if current_name and upper.startswith("*ELASTIC"):
            if i + 1 < len(lines):
                parts = [p.strip() for p in lines[i + 1].split(",")]
                materials[current_name]["E"] = float(parts[0])
                materials[current_name]["nu"] = float(parts[1])
            i += 2
            continue
        if current_name and upper.startswith("*PLASTIC"):
            table_rows: list[list[float]] = []
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                if not row or row.startswith("**"):
                    i += 1
                    continue
                if row.startswith("*"):
                    break
                parts = [p.strip() for p in row.split(",") if p.strip()]
                values = [float(item) for item in parts]
                if values:
                    table_rows.append(values)
                i += 1

            if table_rows:
                table = np.asarray(table_rows, dtype=np.float64)
                materials[current_name]["plastic_table"] = table
                materials[current_name]["yield_stress"] = float(table[0, 0])
                hardening = "perfect"
                if table.shape[0] > 1 and not np.allclose(table[:, 0], table[0, 0]):
                    hardening = "tabulated"
                materials[current_name]["hardening"] = hardening
            continue
        if upper.startswith("*") and not upper.startswith("**"):
            current_name = None
        i += 1
    return materials


def parse_static_steps(inp_text: str) -> list[dict]:
    """Parse minimal `*Step` / `*Static` metadata for solid workflows."""
    steps: list[dict] = []
    lines = inp_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()
        if not upper.startswith("*STEP"):
            i += 1
            continue

        name_match = re.search(r"NAME\s*=\s*([^,\s]+)", line, re.IGNORECASE)
        nlgeom_match = re.search(r"NLGEOM\s*=\s*([^,\s]+)", line, re.IGNORECASE)
        step_info = {
            "name": name_match.group(1) if name_match else f"STEP_{len(steps) + 1}",
            "procedure": "",
            "nlgeom": (nlgeom_match.group(1).strip().upper() == "YES") if nlgeom_match else False,
            "initial_increment": np.nan,
            "total_time": np.nan,
            "min_increment": np.nan,
            "max_increment": np.nan,
        }

        i += 1
        while i < len(lines):
            row = lines[i].strip()
            row_upper = row.upper()
            if row_upper.startswith("*END STEP"):
                break
            if not row or row.startswith("**"):
                i += 1
                continue
            if row_upper.startswith("*STATIC"):
                step_info["procedure"] = "static"
                if i + 1 < len(lines):
                    data_line = lines[i + 1].strip()
                    if data_line and not data_line.startswith("*"):
                        parts = [p.strip() for p in data_line.split(",") if p.strip()]
                        values = [float(item) for item in parts]
                        if len(values) >= 1:
                            step_info["initial_increment"] = values[0]
                        if len(values) >= 2:
                            step_info["total_time"] = values[1]
                        if len(values) >= 3:
                            step_info["min_increment"] = values[2]
                        if len(values) >= 4:
                            step_info["max_increment"] = values[3]
                        i += 2
                        continue
            i += 1

        steps.append(step_info)
        i += 1

    return steps


def parse_sections(inp_text: str) -> list[dict]:
    sections = []
    for line in inp_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        if stripped.upper().startswith("*SOLID SECTION"):
            elset = re.search(r"ELSET\s*=\s*([^,\s]+)", stripped, re.IGNORECASE)
            material = re.search(r"MATERIAL\s*=\s*([^,\s]+)", stripped, re.IGNORECASE)
            sections.append(
                {
                    "elset": elset.group(1).upper() if elset else "",
                    "material": material.group(1).upper() if material else "",
                }
            )
    return sections


def parse_boundary_blocks(inp_text: str) -> list[dict]:
    blocks = []
    current_name = None
    in_boundary = False
    for line in inp_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("**") and "NAME:" in stripped.upper() and "DISPLACEMENT/ROTATION" in stripped.upper():
            match = re.search(r"NAME:\s*([^,\n]+)", stripped, re.IGNORECASE)
            current_name = match.group(1).strip() if match else None
            continue
        upper = stripped.upper()
        if upper.startswith("*BOUNDARY"):
            in_boundary = True
            continue
        if upper.startswith("*") and not upper.startswith("**"):
            in_boundary = False
            continue
        if in_boundary:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 3:
                blocks.append(
                    {
                        "name": current_name or "",
                        "region": parts[0].upper(),
                        "dof_start": int(parts[1]),
                        "dof_end": int(parts[2]),
                        "value": float(parts[3]) if len(parts) >= 4 and parts[3] else 0.0,
                    }
                )
    return blocks


def parse_dsload(inp_text: str) -> list[dict]:
    results = []
    in_dsload = False
    for line in inp_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        upper = stripped.upper()
        if upper.startswith("*DSLOAD") or upper.startswith("*DLOAD"):
            in_dsload = True
            continue
        if upper.startswith("*") and not upper.startswith("**"):
            in_dsload = False
            continue
        if in_dsload:
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 3:
                results.append(
                    {
                        "surface": parts[0].upper(),
                        "load_type": parts[1].upper(),
                        "value": float(parts[2]),
                    }
                )
    return results


def parse_surface_definitions(inp_text: str) -> dict[str, list[tuple[str, str]]]:
    surfaces = {}
    current_name = None
    current_entries = None
    for line in inp_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        upper = stripped.upper()
        if upper.startswith("*SURFACE"):
            current_name = None
            current_entries = None
            if "TYPE=ELEMENT" not in upper:
                continue
            match = re.search(r"NAME\s*=\s*([^,\s]+)", stripped, re.IGNORECASE)
            if match:
                current_name = match.group(1).upper()
                current_entries = []
                surfaces[current_name] = current_entries
            continue
        if upper.startswith("*") and not upper.startswith("**"):
            current_name = None
            current_entries = None
            continue
        if current_entries is not None:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
            if len(parts) >= 2:
                current_entries.append((parts[0].upper(), parts[1].upper()))
    return surfaces


def parse_element_block_types(inp_text: str) -> list[str]:
    block_types = []
    for line in inp_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("**"):
            continue
        upper = stripped.upper()
        if upper.startswith("*ELEMENT OUTPUT"):
            continue
        if upper.startswith("*ELEMENT"):
            match = re.search(r"TYPE\s*=\s*([^,\s]+)", stripped, re.IGNORECASE)
            if match:
                block_types.append(match.group(1).upper())
    return block_types


def _object_column(items: list) -> np.ndarray:
    arr = np.empty((len(items), 1), dtype=object)
    for idx, item in enumerate(items):
        arr[idx, 0] = item
    return arr


def _collect_surface_node_sets(
    surface_defs: dict[str, list[tuple[str, str]]],
    element_set_blocks: dict[str, list[dict]],
    cell_blocks: list[dict],
) -> dict[str, np.ndarray]:
    surface_nodes = {}
    for surf_name, items in surface_defs.items():
        node_ids = set()
        for elset_name, face_label in items:
            for record in element_set_blocks.get(elset_name.upper(), []):
                block_id = int(record["block_id"])
                ele_type = str(record["ele_type"]).upper()
                face_nodes = ABAQUS_FACE_INDS_BY_ELE_TYPE.get(ele_type, {}).get(face_label.upper())
                if face_nodes is None:
                    continue
                cells = np.asarray(cell_blocks[block_id]["cells"], dtype=np.int32)
                for elem_id in np.asarray(record["elem_ids"], dtype=np.int32).reshape(-1):
                    conn = cells[int(elem_id)]
                    node_ids.update(int(node_id) for node_id in conn[face_nodes])
        surface_nodes[surf_name.upper()] = np.asarray(sorted(node_ids), dtype=np.int32)
    return surface_nodes


def parse_inp_model(inp_path: str) -> dict:
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is required for INP parsing. "
            "Install it in WSL: pip install meshio"
        )

    inp_path = str(Path(inp_path).resolve())
    mesh = meshio.read(inp_path)
    points = np.asarray(mesh.points, dtype=np.float64)
    inp_text = _read_inp_text(inp_path)

    block_ele_types = parse_element_block_types(inp_text)
    if block_ele_types and len(block_ele_types) != len(mesh.cells):
        raise ValueError(
            f"INP element block count ({len(block_ele_types)}) does not match "
            f"meshio cell block count ({len(mesh.cells)})."
        )

    cell_blocks = []
    cells_by_type_parts: dict[str, list[np.ndarray]] = {}
    cell_type_offsets_by_block = []
    counts_by_type: dict[str, int] = {}
    for block_id, cell_block in enumerate(mesh.cells):
        cell_type = str(cell_block.type)
        cells = np.asarray(cell_block.data, dtype=np.int32)
        ele_type = (
            block_ele_types[block_id]
            if block_id < len(block_ele_types)
            else DEFAULT_ABAQUS_ELE_TYPE_BY_CELL_TYPE.get(cell_type, cell_type.upper())
        )
        expected_cell_type = ABAQUS_TO_MESHIO_CELL_TYPE.get(ele_type)
        if expected_cell_type is not None and expected_cell_type != cell_type:
            raise ValueError(
                f"Element block {block_id} parsed as ABAQUS {ele_type} but meshio cell type is "
                f"{cell_type} (expected {expected_cell_type})."
            )
        offset = counts_by_type.get(cell_type, 0)
        cell_type_offsets_by_block.append(offset)
        counts_by_type[cell_type] = offset + int(cells.shape[0])
        cells_by_type_parts.setdefault(cell_type, []).append(cells)
        cell_blocks.append(
            {
                "block_id": block_id,
                "cell_type": cell_type,
                "ele_type": ele_type,
                "cells": cells,
            }
        )

    cell_type_names = sorted(cells_by_type_parts.keys())
    if not cell_type_names:
        raise ValueError("No element blocks found in INP.")
    cells_by_type = {
        cell_type: (
            np.asarray(parts[0], dtype=np.int32)
            if len(parts) == 1
            else np.concatenate(parts, axis=0).astype(np.int32, copy=False)
        )
        for cell_type, parts in cells_by_type_parts.items()
    }

    cell_block_cell_types = [record["cell_type"] for record in cell_blocks]
    cell_block_ele_types = [record["ele_type"] for record in cell_blocks]

    node_sets = {key.upper(): np.asarray(val, dtype=np.int32) for key, val in mesh.point_sets.items()}

    element_sets: dict[str, dict[str, np.ndarray]] = {}
    element_set_block_records = []
    for set_name, block_entries in mesh.cell_sets.items():
        set_key = str(set_name).upper()
        for block_id, elem_ids in enumerate(block_entries):
            elem_ids = np.asarray(elem_ids, dtype=np.int32).reshape(-1)
            if elem_ids.size == 0:
                continue
            block_info = cell_blocks[block_id]
            cell_type = str(block_info["cell_type"])
            ele_type = str(block_info["ele_type"])
            offset = int(cell_type_offsets_by_block[block_id])
            element_sets.setdefault(set_key, {}).setdefault(cell_type, []).append(elem_ids + offset)
            element_set_block_records.append(
                {
                    "name": set_key,
                    "block_id": block_id,
                    "cell_type": cell_type,
                    "ele_type": ele_type,
                    "elem_ids": elem_ids,
                }
            )
    for set_name, type_map in list(element_sets.items()):
        element_sets[set_name] = {
            cell_type: np.concatenate(parts, axis=0).astype(np.int32, copy=False)
            for cell_type, parts in type_map.items()
        }

    element_set_blocks: dict[str, list[dict]] = {}
    for record in element_set_block_records:
        element_set_blocks.setdefault(record["name"], []).append(record)

    element_set_record_names = []
    element_set_record_types = []
    element_set_record_elems = []
    for set_name in sorted(element_sets.keys()):
        for cell_type in sorted(element_sets[set_name].keys()):
            element_set_record_names.append(set_name)
            element_set_record_types.append(cell_type)
            element_set_record_elems.append(element_sets[set_name][cell_type].reshape(1, -1))

    element_set_block_record_names = []
    element_set_block_record_ids = []
    element_set_block_record_types = []
    element_set_block_record_ele_types = []
    element_set_block_record_elems = []
    for record in element_set_block_records:
        element_set_block_record_names.append(record["name"])
        element_set_block_record_ids.append(int(record["block_id"]))
        element_set_block_record_types.append(record["cell_type"])
        element_set_block_record_ele_types.append(record["ele_type"])
        element_set_block_record_elems.append(np.asarray(record["elem_ids"], dtype=np.int32).reshape(1, -1))

    materials = parse_materials(inp_text)
    sections = parse_sections(inp_text)
    boundaries = parse_boundary_blocks(inp_text)
    surface_defs = parse_surface_definitions(inp_text)
    dsloads = parse_dsload(inp_text)
    static_steps = parse_static_steps(inp_text)
    surface_node_sets = _collect_surface_node_sets(surface_defs, element_set_blocks, cell_blocks)

    material_names = sorted(materials.keys())
    if not material_names:
        raise ValueError("No *Material/*Elastic definition found in INP.")
    if len(material_names) > 1:
        raise NotImplementedError(
            "Current solid parser is limited to the elastic path with a single isotropic material. "
            "Future nonlinear workflows must extend the parser and merge .inp data with .for data."
        )

    mat_name = material_names[0]
    E = float(materials[mat_name]["E"])
    nu = float(materials[mat_name]["nu"])
    has_plastic = any("plastic_table" in materials[name] for name in material_names)

    step_names = [step["name"] for step in static_steps]
    step_procedures = [str(step["procedure"]) for step in static_steps]
    step_initial_increment = np.asarray(
        [[float(step["initial_increment"])] for step in static_steps],
        dtype=np.float64,
    )
    step_total_time = np.asarray(
        [[float(step["total_time"])] for step in static_steps],
        dtype=np.float64,
    )
    step_min_increment = np.asarray(
        [[float(step["min_increment"])] for step in static_steps],
        dtype=np.float64,
    )
    step_max_increment = np.asarray(
        [[float(step["max_increment"])] for step in static_steps],
        dtype=np.float64,
    )
    step_nlgeom = np.asarray(
        [[int(bool(step["nlgeom"]))] for step in static_steps],
        dtype=np.int32,
    )

    return {
        "created_at": np.asarray([dt.datetime.now().isoformat()]),
        "inp_path": np.asarray([inp_path]),
        "parser_scope": np.asarray(["solid_static_inp" if has_plastic or static_steps else "elastic_inp_only"]),
        "points": points,
        "cell_type_names": _object_column(cell_type_names),
        "cell_type_cells": _object_column([cells_by_type[name] for name in cell_type_names]),
        "cell_block_cell_types": _object_column(cell_block_cell_types),
        "cell_block_ele_types": _object_column(cell_block_ele_types),
        "cell_block_cells": _object_column([record["cells"] for record in cell_blocks]),
        "node_set_names": _object_column(sorted(node_sets.keys())),
        "node_set_nodes": _object_column([node_sets[key].reshape(1, -1) for key in sorted(node_sets.keys())]),
        "element_set_names": _object_column(sorted(element_sets.keys())),
        "element_set_elems": _object_column(
            [
                element_sets[key]["hexahedron"].reshape(1, -1)
                if "hexahedron" in element_sets[key]
                else np.zeros((1, 0), dtype=np.int32)
                for key in sorted(element_sets.keys())
            ]
        ),
        "element_set_record_names": _object_column(element_set_record_names),
        "element_set_record_types": _object_column(element_set_record_types),
        "element_set_record_elems": _object_column(element_set_record_elems),
        "element_set_block_record_names": _object_column(element_set_block_record_names),
        "element_set_block_record_ids": np.asarray(element_set_block_record_ids, dtype=np.int32).reshape(-1, 1),
        "element_set_block_record_types": _object_column(element_set_block_record_types),
        "element_set_block_record_etype": _object_column(element_set_block_record_ele_types),
        "element_set_block_record_elems": _object_column(element_set_block_record_elems),
        "material_names": _object_column(material_names),
        "material_E": np.asarray([[E]], dtype=np.float64),
        "material_nu": np.asarray([[nu]], dtype=np.float64),
        "material_yield_stress": np.asarray(
            [[float(materials[name].get("yield_stress", np.nan))] for name in material_names],
            dtype=np.float64,
        ),
        "material_hardening": _object_column([str(materials[name].get("hardening", "")) for name in material_names]),
        "material_plastic_tables": _object_column(
            [
                np.asarray(materials[name].get("plastic_table", np.zeros((0, 2), dtype=np.float64)), dtype=np.float64)
                for name in material_names
            ]
        ),
        "section_elsets": _object_column([sec["elset"] for sec in sections]),
        "section_materials": _object_column([sec["material"] for sec in sections]),
        "step_names": _object_column(step_names),
        "step_procedures": _object_column(step_procedures),
        "step_initial_increment": step_initial_increment,
        "step_total_time": step_total_time,
        "step_min_increment": step_min_increment,
        "step_max_increment": step_max_increment,
        "step_nlgeom": step_nlgeom,
        "boundary_names": _object_column([bc["name"] for bc in boundaries]),
        "boundary_regions": _object_column([bc["region"] for bc in boundaries]),
        "boundary_dof_start": np.asarray([[bc["dof_start"]] for bc in boundaries], dtype=np.int32),
        "boundary_dof_end": np.asarray([[bc["dof_end"]] for bc in boundaries], dtype=np.int32),
        "boundary_value": np.asarray([[bc["value"]] for bc in boundaries], dtype=np.float64),
        "surface_names": _object_column(list(surface_defs.keys())),
        "surface_items": _object_column([surface_defs[name] for name in surface_defs.keys()]),
        "dsload_surfaces": _object_column([item["surface"] for item in dsloads]),
        "dsload_types": _object_column([item["load_type"] for item in dsloads]),
        "dsload_values": np.asarray([[item["value"]] for item in dsloads], dtype=np.float64),
        "dsload_surface_nodes": _object_column(
            [
                surface_node_sets.get(item["surface"].upper(), np.zeros((0,), dtype=np.int32)).reshape(1, -1)
                for item in dsloads
            ]
        ),
    }


def save_parsed_model_mat(model_data: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat_payload = {
        "InpData": {
            **model_data,
            "info": np.asarray(
                [
                    "InpData stores the current elastic-model inputs parsed from ABAQUS INP: "
                    "points, cell blocks, node and element sets, materials, sections, boundary "
                    "conditions, surface definitions, dsload records, and the node sets attached "
                    "to each dsload surface. Future nonlinear workflows will require richer data "
                    "merged from both INP and FOR sources."
                ]
            ),
        }
    }
    sio.savemat(out_path, mat_payload, do_compression=True)
    return out_path


def load_parsed_model_mat(mat_path: str | Path) -> dict:
    raw = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    source = raw["InpData"][0, 0] if "InpData" in raw else raw
    cell_type_names = [str(np.asarray(x).reshape(-1)[0]) for x in source.cell_type_names.reshape(-1)]
    cells_by_type = {
        name: np.asarray(source.cell_type_cells.reshape(-1)[idx], dtype=np.int32)
        for idx, name in enumerate(cell_type_names)
    }
    node_set_names = [str(np.asarray(x).reshape(-1)[0]) for x in source.node_set_names.reshape(-1)]
    node_set_nodes = {
        name: np.asarray(source.node_set_nodes.reshape(-1)[idx], dtype=np.int32).reshape(-1)
        for idx, name in enumerate(node_set_names)
    }
    if hasattr(source, "cell_block_cell_types"):
        cell_block_cell_types = [str(np.asarray(x).reshape(-1)[0]) for x in source.cell_block_cell_types.reshape(-1)]
        cell_block_ele_types = [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.cell_block_ele_types.reshape(-1)]
        cell_block_cells = source.cell_block_cells.reshape(-1)
        cell_blocks = [
            {
                "block_id": idx,
                "cell_type": cell_block_cell_types[idx],
                "ele_type": cell_block_ele_types[idx],
                "cells": np.asarray(cell_block_cells[idx], dtype=np.int32),
            }
            for idx in range(len(cell_block_cell_types))
        ]
    else:
        cell_blocks = [
            {
                "block_id": idx,
                "cell_type": name,
                "ele_type": DEFAULT_ABAQUS_ELE_TYPE_BY_CELL_TYPE.get(name, name.upper()),
                "cells": np.asarray(source.cell_type_cells.reshape(-1)[idx], dtype=np.int32),
            }
            for idx, name in enumerate(cell_type_names)
        ]
    if hasattr(source, "element_set_record_names"):
        element_set_elems = {}
        record_names = [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.element_set_record_names.reshape(-1)]
        record_types = [str(np.asarray(x).reshape(-1)[0]) for x in source.element_set_record_types.reshape(-1)]
        record_elems = source.element_set_record_elems.reshape(-1)
        for set_name, cell_type, elems in zip(record_names, record_types, record_elems):
            element_set_elems.setdefault(set_name, {})[cell_type] = np.asarray(elems, dtype=np.int32).reshape(-1)
    else:
        element_set_names = [str(np.asarray(x).reshape(-1)[0]) for x in source.element_set_names.reshape(-1)]
        element_set_elems = {
            name: {"hexahedron": np.asarray(source.element_set_elems.reshape(-1)[idx], dtype=np.int32).reshape(-1)}
            for idx, name in enumerate(element_set_names)
        }
    if hasattr(source, "element_set_block_record_names"):
        element_set_blocks = {}
        record_names = [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.element_set_block_record_names.reshape(-1)]
        record_ids = np.asarray(source.element_set_block_record_ids, dtype=np.int32).reshape(-1)
        record_types = [str(np.asarray(x).reshape(-1)[0]) for x in source.element_set_block_record_types.reshape(-1)]
        record_ele_types = [
            str(np.asarray(x).reshape(-1)[0]).upper()
            for x in (
                source.element_set_block_record_etype.reshape(-1)
                if hasattr(source, "element_set_block_record_etype")
                else source.element_set_block_record_ele_types.reshape(-1)
            )
        ]
        record_elems = source.element_set_block_record_elems.reshape(-1)
        for set_name, block_id, cell_type, ele_type, elems in zip(
            record_names, record_ids, record_types, record_ele_types, record_elems
        ):
            element_set_blocks.setdefault(set_name, []).append(
                {
                    "block_id": int(block_id),
                    "cell_type": cell_type,
                    "ele_type": ele_type,
                    "elem_ids": np.asarray(elems, dtype=np.int32).reshape(-1),
                }
            )
    else:
        element_set_blocks = {}
        for set_name, type_map in element_set_elems.items():
            for block in cell_blocks:
                cell_type = str(block["cell_type"])
                if cell_type not in type_map:
                    continue
                element_set_blocks.setdefault(set_name, []).append(
                    {
                        "block_id": int(block["block_id"]),
                        "cell_type": cell_type,
                        "ele_type": str(block["ele_type"]).upper(),
                        "elem_ids": np.asarray(type_map[cell_type], dtype=np.int32).reshape(-1),
                    }
                )
    boundaries = []
    for idx in range(source.boundary_regions.shape[0]):
        boundaries.append(
            {
                "name": str(np.asarray(source.boundary_names[idx, 0]).reshape(-1)[0]),
                "region": str(np.asarray(source.boundary_regions[idx, 0]).reshape(-1)[0]).upper(),
                "dof_start": int(np.asarray(source.boundary_dof_start[idx, 0]).reshape(-1)[0]),
                "dof_end": int(np.asarray(source.boundary_dof_end[idx, 0]).reshape(-1)[0]),
                "value": float(np.asarray(source.boundary_value[idx, 0]).reshape(-1)[0]),
            }
        )
    surface_defs = {}
    if hasattr(source, "surface_names"):
        names = [str(np.asarray(x).reshape(-1)[0]) for x in source.surface_names.reshape(-1)]
        items = source.surface_items.reshape(-1)
        for idx, name in enumerate(names):
            cleaned = []
            for rec in list(items[idx]):
                arr = np.asarray(rec).reshape(-1)
                if arr.size >= 2:
                    cleaned.append((str(arr[0]).strip().upper(), str(arr[1]).strip().upper()))
            surface_defs[name.upper()] = cleaned
    dsloads = []
    if hasattr(source, "dsload_surfaces"):
        for idx in range(source.dsload_surfaces.shape[0]):
            dsloads.append(
                {
                    "surface": str(np.asarray(source.dsload_surfaces[idx, 0]).reshape(-1)[0]).upper(),
                    "load_type": str(np.asarray(source.dsload_types[idx, 0]).reshape(-1)[0]).upper(),
                    "value": float(np.asarray(source.dsload_values[idx, 0]).reshape(-1)[0]),
                    "nodes": (
                        np.asarray(source.dsload_surface_nodes.reshape(-1)[idx], dtype=np.int32).reshape(-1)
                        if hasattr(source, "dsload_surface_nodes")
                        else np.zeros((0,), dtype=np.int32)
                    ),
                }
            )
    return {
        "inp_path": str(np.asarray(source.inp_path).reshape(-1)[0]),
        "points": np.asarray(source.points, dtype=np.float64),
        "cell_types": cell_type_names,
        "cells_by_type": cells_by_type,
        "cell_blocks": cell_blocks,
        "node_sets": node_set_nodes,
        "element_sets": element_set_elems,
        "element_set_blocks": element_set_blocks,
        "material_names": [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.material_names.reshape(-1)],
        "E": float(np.asarray(source.material_E).reshape(-1)[0]),
        "nu": float(np.asarray(source.material_nu).reshape(-1)[0]),
        "section_elsets": [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.section_elsets.reshape(-1)],
        "section_materials": [str(np.asarray(x).reshape(-1)[0]).upper() for x in source.section_materials.reshape(-1)],
        "boundaries": boundaries,
        "surface_defs": surface_defs,
        "dsloads": dsloads,
        "metadata": {
            "created_at": str(np.asarray(source.created_at).reshape(-1)[0]),
            "parser_scope": (
                str(np.asarray(source.parser_scope).reshape(-1)[0])
                if hasattr(source, "parser_scope")
                else "elastic_inp_only"
            ),
            "info": str(np.asarray(source.info).reshape(-1)[0]) if hasattr(source, "info") else "",
        },
        "plastic": {
            "yield_stress": (
                float(np.asarray(source.material_yield_stress).reshape(-1)[0])
                if hasattr(source, "material_yield_stress") and np.asarray(source.material_yield_stress).size
                else None
            ),
            "hardening": (
                str(np.asarray(source.material_hardening).reshape(-1)[0])
                if hasattr(source, "material_hardening") and np.asarray(source.material_hardening).size
                else ""
            ),
            "table": (
                np.asarray(source.material_plastic_tables.reshape(-1)[0], dtype=np.float64)
                if hasattr(source, "material_plastic_tables") and source.material_plastic_tables.size
                else np.zeros((0, 2), dtype=np.float64)
            ),
        },
        "steps": [
            {
                "name": str(np.asarray(source.step_names.reshape(-1)[idx]).reshape(-1)[0]),
                "procedure": str(np.asarray(source.step_procedures.reshape(-1)[idx]).reshape(-1)[0]),
                "initial_increment": float(np.asarray(source.step_initial_increment).reshape(-1)[idx]),
                "total_time": float(np.asarray(source.step_total_time).reshape(-1)[idx]),
                "min_increment": float(np.asarray(source.step_min_increment).reshape(-1)[idx]),
                "max_increment": float(np.asarray(source.step_max_increment).reshape(-1)[idx]),
                "nlgeom": bool(int(np.asarray(source.step_nlgeom).reshape(-1)[idx])),
            }
            for idx in range(len(source.step_names.reshape(-1)))
        ] if hasattr(source, "step_names") else [],
    }
