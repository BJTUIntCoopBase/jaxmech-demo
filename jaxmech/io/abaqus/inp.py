"""
jaxmech.io.abaqus.inp — Unified ABAQUS INP parser.

This module provides a single entry point ``parse_inp()`` that:
  1. Detects whether the INP contains solid or shell elements.
  2. Dispatches to the appropriate existing parser.
  3. Converts the result into a unified ``jaxmech.model.Model`` object.

Currently uses native parsers under ``jaxmech.io.abaqus`` for both solid and
shell models.

The current solid parser is intentionally scoped to elastic models. More
complex future workflows will need richer parsing and a merged input model
built from both ``.inp`` and companion ``.for`` files.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Optional

import numpy as np

from jaxmech.fem.mesh import Mesh, ElementBlock
from jaxmech.fem.section import SolidSection, ShellSection
from jaxmech.materials.definition import MaterialDef
from jaxmech.model.model import Model
from jaxmech.model.loads import LoadCase, SurfaceLoad, ConcentratedLoad
from jaxmech.model.boundary import BoundaryCondition

try:
    from jaxmech.io.abaqus.shell_inp import parse_shell_inp
    _HAS_SHELL_INP = True
except ImportError:
    _HAS_SHELL_INP = False


# ─── Element type classification ───────────────────────────────────────────

SHELL_ELE_TYPES = {"S4", "S4R", "S3", "S3R", "STRI3", "STRI65"}
SOLID_ELE_TYPES = {
    "C3D8", "C3D8R", "C3D6", "C3D4",
    "C3D20", "C3D20R", "C3D10",
    "CPS4", "CPS4R", "CPS3",
    "CPE4", "CPE4R", "CPE3",
}

# meshio cell_type → ABAQUS ele_type fallback
ABAQUS_TO_MESHIO_CELL_TYPE = {
    "CPS4": "quad", "CPS4R": "quad", "CPS3": "triangle",
    "C3D8": "hexahedron", "C3D8R": "hexahedron",
    "C3D6": "wedge", "C3D4": "tetra",
    "STRI3": "triangle", "S3": "triangle",
    "S4": "quad", "S4R": "quad",
}


def _normalize_inp_path(inp_path: str | Path) -> str:
    text = str(inp_path)
    if len(text) >= 3 and text[1:3] == ":\\" and os.name != "nt":
        drive = text[0].lower()
        tail = text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        if os.name != "nt":
            return text
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        return f"{drive}:\\{tail}"
    return str(Path(text).resolve())


def _detect_element_family(inp_path: str | Path) -> str:
    """Scan the INP file to determine whether it contains shell or solid elements.

    Returns
    -------
    "shell" or "solid"
    """
    with open(inp_path, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            s = line.strip().upper()
            if s.startswith("*ELEMENT") and not s.startswith("*ELEMENT OUTPUT") \
               and not s.startswith("*ELSET"):
                m = re.search(r"TYPE\s*=\s*([^,\s]+)", s)
                if m:
                    ele_type = m.group(1).upper()
                    if ele_type in SHELL_ELE_TYPES:
                        return "shell"
                    if ele_type in SOLID_ELE_TYPES:
                        return "solid"
    return "solid"  # default


# ─── Main entry point ──────────────────────────────────────────────────────

def parse_inp(inp_path: str | Path) -> Model:
    """Parse an ABAQUS INP file and return a unified ``Model`` object.

    Parameters
    ----------
    inp_path : str or Path
        Path to the .inp file.

    Returns
    -------
    Model
        Unified finite element model.
    """
    inp_path = _normalize_inp_path(inp_path)
    family = _detect_element_family(inp_path)

    if family == "shell":
        if not _HAS_SHELL_INP:
            raise NotImplementedError(
                "Shell element parsing is not available in this demo. "
                "Use the full jaxmech version for shell support."
            )
        return _parse_shell(inp_path)
    else:
        return _parse_solid(inp_path)


# ─── Solid conversion ─────────────────────────────────────────────────────

def _parse_solid(inp_path: str) -> Model:
    """Dispatch to the native solid parser and convert to Model."""
    from jaxmech.io.abaqus.solid_inp import parse_inp_model

    raw = parse_inp_model(inp_path)

    # --- Mesh ---
    points = raw["points"]

    cell_block_cell_types = [
        str(np.asarray(x).reshape(-1)[0])
        for x in raw["cell_block_cell_types"].reshape(-1)
    ]
    cell_block_ele_types = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["cell_block_ele_types"].reshape(-1)
    ]
    cell_block_cells = raw["cell_block_cells"].reshape(-1)

    blocks = []
    for i, (ct, et) in enumerate(zip(cell_block_cell_types, cell_block_ele_types)):
        conn = np.asarray(cell_block_cells[i], dtype=np.int32)
        blocks.append(ElementBlock(
            block_id=i,
            cell_type=ct,
            ele_type=et,
            connectivity=conn,
        ))

    # Node sets
    node_set_names = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["node_set_names"].reshape(-1)
    ]
    node_set_nodes = raw["node_set_nodes"].reshape(-1)
    node_sets = {
        name: np.asarray(node_set_nodes[i], dtype=np.int32).reshape(-1)
        for i, name in enumerate(node_set_names)
    }

    mesh = Mesh(
        nodes=points,
        blocks=blocks,
        node_sets=node_sets,
        metadata={
            "source_file": inp_path,
            "parser": "solid",
            "_solid_inp_data": raw,
        },
    )

    # --- Materials ---
    mat_names = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["material_names"].reshape(-1)
    ]
    E = float(np.asarray(raw["material_E"]).reshape(-1)[0])
    nu = float(np.asarray(raw["material_nu"]).reshape(-1)[0])
    plastic_tables = raw["material_plastic_tables"].reshape(-1) if "material_plastic_tables" in raw else []
    yield_values = np.asarray(raw["material_yield_stress"], dtype=np.float64).reshape(-1) if "material_yield_stress" in raw else np.asarray([], dtype=np.float64)
    hardening_types = raw["material_hardening"].reshape(-1) if "material_hardening" in raw else []
    materials: list[MaterialDef] = []
    for idx, name in enumerate(mat_names):
        plastic_payload = None
        if idx < len(plastic_tables):
            table = np.asarray(plastic_tables[idx], dtype=np.float64)
            if table.size:
                plastic_payload = {
                    "table": table,
                    "yield_stress": float(yield_values[idx]) if idx < len(yield_values) and np.isfinite(yield_values[idx]) else float(table.reshape(-1)[0]),
                    "hardening": str(np.asarray(hardening_types[idx]).reshape(-1)[0]) if idx < len(hardening_types) else "",
                }
        materials.append(
            MaterialDef(
                name=name,
                elastic={"E": E, "nu": nu, "type": "isotropic"},
                plastic=plastic_payload,
            )
        )

    # --- Sections ---
    sec_elsets = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["section_elsets"].reshape(-1)
    ]
    sec_materials = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["section_materials"].reshape(-1)
    ]
    sections = [
        SolidSection(
            name=f"section_{i}",
            material_name=mat,
            elset_name=es,
        )
        for i, (es, mat) in enumerate(zip(sec_elsets, sec_materials))
    ]

    # --- Boundary conditions ---
    bc_regions = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["boundary_regions"].reshape(-1)
    ]
    bc_dof_start = np.asarray(raw["boundary_dof_start"], dtype=np.int32).reshape(-1)
    bc_dof_end = np.asarray(raw["boundary_dof_end"], dtype=np.int32).reshape(-1)
    bc_values = np.asarray(raw["boundary_value"], dtype=np.float64).reshape(-1)
    bcs = [
        BoundaryCondition(
            node_set=region,
            dof_first=int(bc_dof_start[i]),
            dof_last=int(bc_dof_end[i]),
            value=float(bc_values[i]),
        )
        for i, region in enumerate(bc_regions)
    ]

    # --- Load cases ---
    surface_loads = []
    dsload_surfaces = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["dsload_surfaces"].reshape(-1)
    ]
    dsload_types = [
        str(np.asarray(x).reshape(-1)[0]).upper()
        for x in raw["dsload_types"].reshape(-1)
    ]
    dsload_values = np.asarray(raw["dsload_values"], dtype=np.float64).reshape(-1)
    for i in range(len(dsload_surfaces)):
        surface_loads.append(SurfaceLoad(
            surface_name=dsload_surfaces[i],
            load_type=dsload_types[i],
            magnitude=float(dsload_values[i]),
        ))

    step_metadata: dict[str, object] = {}
    if "step_names" in raw and np.asarray(raw["step_names"]).size:
        step_names = raw["step_names"].reshape(-1)
        step_metadata = {
            "step_name": str(np.asarray(step_names[0]).reshape(-1)[0]),
            "procedure": str(np.asarray(raw["step_procedures"].reshape(-1)[0]).reshape(-1)[0]) if "step_procedures" in raw else "",
            "initial_increment": float(np.asarray(raw["step_initial_increment"], dtype=np.float64).reshape(-1)[0]) if "step_initial_increment" in raw else np.nan,
            "total_time": float(np.asarray(raw["step_total_time"], dtype=np.float64).reshape(-1)[0]) if "step_total_time" in raw else np.nan,
            "min_increment": float(np.asarray(raw["step_min_increment"], dtype=np.float64).reshape(-1)[0]) if "step_min_increment" in raw else np.nan,
            "max_increment": float(np.asarray(raw["step_max_increment"], dtype=np.float64).reshape(-1)[0]) if "step_max_increment" in raw else np.nan,
            "nlgeom": bool(int(np.asarray(raw["step_nlgeom"], dtype=np.int32).reshape(-1)[0])) if "step_nlgeom" in raw else False,
        }

    load_case = LoadCase(
        name=str(step_metadata.get("step_name", Path(inp_path).stem)),
        surface_loads=surface_loads,
        metadata=step_metadata,
    )

    return Model(
        mesh=mesh,
        materials=materials,
        sections=sections,
        load_cases=[load_case],
        bcs=bcs,
        metadata={
            "source_file": inp_path,
            "family": "solid",
            "dimension": 3 if points.shape[1] == 3 else 2,
            "input_mode": "parsed_only",
            "parser_scope": str(np.asarray(raw["parser_scope"]).reshape(-1)[0]) if "parser_scope" in raw else "elastic_inp_only",
            "step_metadata": step_metadata,
            "future_input_note": (
                "Current solid parser only targets elastic INP data. "
                "Future nonlinear workflows will require richer parsing and "
                "merging INP data with FOR data."
            ),
            "_solid_inp_data": raw,
        },
    )


# ─── Shell conversion ─────────────────────────────────────────────────────

def shell_data_to_model(data) -> Model:
    """Convert parsed ``ShellInpData`` to the unified ``Model`` object."""
    # --- Mesh ---
    cell_type = ABAQUS_TO_MESHIO_CELL_TYPE.get(data.ele_type, "triangle")
    blocks = [ElementBlock(
        block_id=0,
        cell_type=cell_type,
        ele_type=data.ele_type,
        connectivity=data.elem_conn,
        element_ids=data.elem_labels,
    )]

    # Convert shell node sets: labels -> 0-based indices
    node_sets = {}
    for name, labels in data.nsets.items():
        indices = np.array(
            [data._node_label_to_idx[int(lbl)]
             for lbl in labels if int(lbl) in data._node_label_to_idx],
            dtype=np.int32,
        )
        node_sets[name.upper()] = indices

    mesh = Mesh(
        nodes=data.node_coords,
        blocks=blocks,
        node_ids=data.node_labels,
        node_sets=node_sets,
        metadata={"source_file": data.inp_path, "parser": "shell"},
    )

    materials = [
        MaterialDef(
            name=data.material_name,
            elastic={"E": data.E, "nu": data.nu, "type": "isotropic"},
        )
    ]

    sections = [
        ShellSection(
            name=f"shell_section_{i}",
            material_name=sec.material,
            elset_name=sec.elset,
            thickness=sec.thickness,
            num_int_pts=sec.n_section_points,
        )
        for i, sec in enumerate(data.shell_sections)
    ]

    bcs = [
        BoundaryCondition(
            node_set=bc.nset_or_node.upper(),
            dof_first=bc.dof_start,
            dof_last=bc.dof_end,
            value=bc.value,
        )
        for bc in data.boundaries
    ]

    concentrated = [
        ConcentratedLoad(
            node_set=cl.nset_or_node.upper(),
            dof=int(cl.dof) - 1,
            magnitude=cl.magnitude,
        )
        for cl in data.cloads
    ]
    surface_loads = [
        SurfaceLoad(
            surface_name=ds.surface_name,
            load_type=str(ds.load_type).upper(),
            magnitude=ds.magnitude,
            direction=(
                np.asarray(ds.direction, dtype=np.float64)
                if ds.direction is not None else None
            ),
        )
        for ds in data.dsloads
    ]
    load_case = LoadCase(
        name=Path(data.inp_path).stem,
        concentrated=concentrated,
        surface_loads=surface_loads,
    )

    return Model(
        mesh=mesh,
        materials=materials,
        sections=sections,
        load_cases=[load_case],
        bcs=bcs,
        metadata={
            "source_file": data.inp_path,
            "family": "shell",
            "ele_type": data.ele_type,
            "dimension": 3,
            "_shell_inp_data": data,
        },
    )


def _parse_shell(inp_path: str) -> Model:
    """Dispatch to the existing shell parser and convert to Model."""
    from jaxmech.io.abaqus.shell_inp import parse_shell_inp

    data = parse_shell_inp(inp_path)
    return shell_data_to_model(data)
