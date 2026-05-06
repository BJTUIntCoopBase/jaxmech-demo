"""Solid ABAQUS INP parser entry point for the demo subset."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from jaxmech.fem.mesh import ElementBlock, Mesh
from jaxmech.fem.section import SolidSection
from jaxmech.materials.definition import MaterialDef
from jaxmech.model.boundary import BoundaryCondition
from jaxmech.model.loads import LoadCase, SurfaceLoad
from jaxmech.model.model import Model


SOLID_ELE_TYPES = {
    "C3D8", "C3D8R", "C3D6", "C3D4",
    "C3D20", "C3D20R", "C3D10",
}
SHELL_ELE_TYPES = {"S4", "S4R", "S3", "S3R", "STRI3", "STRI65"}


def _normalize_inp_path(inp_path: str | Path) -> str:
    text = str(inp_path)
    if len(text) >= 3 and text[1:3] == ":\\" and os.name != "nt":
        tail = text[3:].replace("\\", "/")
        return f"/mnt/{text[0].lower()}/{tail}"
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/" and os.name == "nt":
        tail = text[7:].replace("/", "\\")
        return f"{text[5].upper()}:\\{tail}"
    return str(Path(text).resolve())


def _detect_element_family(inp_path: str | Path) -> str:
    with open(inp_path, "r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            text = line.strip().upper()
            if text.startswith("*ELEMENT") and not text.startswith("*ELEMENT OUTPUT") and not text.startswith("*ELSET"):
                match = re.search(r"TYPE\s*=\s*([^,\s]+)", text)
                if not match:
                    continue
                ele_type = match.group(1).upper()
                if ele_type in SHELL_ELE_TYPES:
                    return "shell"
                if ele_type in SOLID_ELE_TYPES:
                    return "solid"
    return "solid"


def parse_inp(inp_path: str | Path) -> Model:
    """Parse a demo-supported solid ABAQUS INP file."""
    inp_path = _normalize_inp_path(inp_path)
    family = _detect_element_family(inp_path)
    if family != "solid":
        raise NotImplementedError("The demo supports solid element INP files only.")
    return _parse_solid(inp_path)


def _text_list(raw: dict, key: str) -> list[str]:
    return [str(np.asarray(item).reshape(-1)[0]).upper() for item in raw[key].reshape(-1)]


def _parse_solid(inp_path: str) -> Model:
    from jaxmech.io.abaqus.solid_inp import parse_inp_model

    raw = parse_inp_model(inp_path)
    points = raw["points"]

    cell_block_cell_types = [str(np.asarray(item).reshape(-1)[0]) for item in raw["cell_block_cell_types"].reshape(-1)]
    cell_block_ele_types = _text_list(raw, "cell_block_ele_types")
    cell_block_cells = raw["cell_block_cells"].reshape(-1)
    blocks = [
        ElementBlock(block_id=i, cell_type=cell_type, ele_type=ele_type, connectivity=np.asarray(cell_block_cells[i], dtype=np.int32))
        for i, (cell_type, ele_type) in enumerate(zip(cell_block_cell_types, cell_block_ele_types))
    ]

    node_set_names = _text_list(raw, "node_set_names")
    node_set_nodes = raw["node_set_nodes"].reshape(-1)
    node_sets = {name: np.asarray(node_set_nodes[i], dtype=np.int32).reshape(-1) for i, name in enumerate(node_set_names)}
    mesh = Mesh(nodes=points, blocks=blocks, node_sets=node_sets, metadata={"source_file": inp_path, "parser": "solid", "_solid_inp_data": raw})

    mat_names = _text_list(raw, "material_names")
    e_values = np.asarray(raw["material_E"], dtype=np.float64).reshape(-1)
    nu_values = np.asarray(raw["material_nu"], dtype=np.float64).reshape(-1)
    materials = [
        MaterialDef(
            name=name,
            elastic={
                "E": float(e_values[min(i, e_values.size - 1)]),
                "nu": float(nu_values[min(i, nu_values.size - 1)]),
                "type": "isotropic",
            },
        )
        for i, name in enumerate(mat_names)
    ]

    sections = [
        SolidSection(name=f"section_{i}", material_name=mat, elset_name=elset)
        for i, (elset, mat) in enumerate(zip(_text_list(raw, "section_elsets"), _text_list(raw, "section_materials")))
    ]

    bc_regions = _text_list(raw, "boundary_regions")
    bc_dof_start = np.asarray(raw["boundary_dof_start"], dtype=np.int32).reshape(-1)
    bc_dof_end = np.asarray(raw["boundary_dof_end"], dtype=np.int32).reshape(-1)
    bc_values = np.asarray(raw["boundary_value"], dtype=np.float64).reshape(-1)
    bcs = [
        BoundaryCondition(node_set=region, dof_first=int(bc_dof_start[i]), dof_last=int(bc_dof_end[i]), value=float(bc_values[i]))
        for i, region in enumerate(bc_regions)
    ]

    surface_loads = []
    dsload_surfaces = _text_list(raw, "dsload_surfaces")
    dsload_types = _text_list(raw, "dsload_types")
    dsload_values = np.asarray(raw["dsload_values"], dtype=np.float64).reshape(-1)
    for i, surface_name in enumerate(dsload_surfaces):
        surface_loads.append(SurfaceLoad(surface_name=surface_name, load_type=dsload_types[i], magnitude=float(dsload_values[i])))

    step_metadata: dict[str, object] = {}
    if "step_names" in raw and np.asarray(raw["step_names"]).size:
        step_metadata = {
            "step_name": str(np.asarray(raw["step_names"].reshape(-1)[0]).reshape(-1)[0]),
            "procedure": str(np.asarray(raw["step_procedures"].reshape(-1)[0]).reshape(-1)[0]) if "step_procedures" in raw else "",
            "initial_increment": float(np.asarray(raw["step_initial_increment"], dtype=np.float64).reshape(-1)[0]) if "step_initial_increment" in raw else np.nan,
            "total_time": float(np.asarray(raw["step_total_time"], dtype=np.float64).reshape(-1)[0]) if "step_total_time" in raw else np.nan,
            "min_increment": float(np.asarray(raw["step_min_increment"], dtype=np.float64).reshape(-1)[0]) if "step_min_increment" in raw else np.nan,
            "max_increment": float(np.asarray(raw["step_max_increment"], dtype=np.float64).reshape(-1)[0]) if "step_max_increment" in raw else np.nan,
            "nlgeom": bool(int(np.asarray(raw["step_nlgeom"], dtype=np.int32).reshape(-1)[0])) if "step_nlgeom" in raw else False,
        }
    load_case = LoadCase(name=str(step_metadata.get("step_name", Path(inp_path).stem)), surface_loads=surface_loads, metadata=step_metadata)

    return Model(
        mesh=mesh,
        materials=materials,
        sections=sections,
        load_cases=[load_case],
        bcs=bcs,
        metadata={
            "source_file": inp_path,
            "family": "solid",
            "dimension": 3,
            "input_mode": "parsed_only",
            "parser_scope": "solid_elastic_inp",
            "step_metadata": step_metadata,
            "_solid_inp_data": raw,
        },
    )
