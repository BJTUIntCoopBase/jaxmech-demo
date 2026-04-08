"""
jaxmech.model.model - Unified finite element model definition.

The Model object is the central data structure that all analysis modules
consume. It aggregates mesh, materials, sections, loads, and boundary
conditions into a single, self-contained object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from jaxmech.fem.mesh import Mesh
from jaxmech.fem.section import Section
from jaxmech.materials.definition import MaterialDef
from jaxmech.model.loads import LoadCase
from jaxmech.model.boundary import BoundaryCondition


@dataclass
class Model:
    """Complete finite element model definition.

    This is the unified data object that downstream modules (elastic solver,
    shakedown, topology optimization, etc.) depend on. It replaces the
    pattern of passing around loose .mat field names.

    Attributes:
        mesh:       The finite element mesh.
        materials:  List of material definitions.
        sections:   List of section assignments.
        load_cases: List of load cases (analysis steps).
        bcs:        List of boundary conditions.
        metadata:   Free-form metadata dict. Suggested keys:
                    - "source_file": path to the original .inp file
                    - "analysis_type": "static", "dynamic", etc.
                    - "dimension": 2 or 3
    """
    mesh: Mesh
    materials: List[MaterialDef] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    load_cases: List[LoadCase] = field(default_factory=list)
    bcs: List[BoundaryCondition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_material(self, name: str) -> Optional[MaterialDef]:
        """Look up a material by name."""
        for m in self.materials:
            if m.name == name:
                return m
        return None

    def get_section(self, name: str) -> Optional[Section]:
        """Look up a section by name."""
        for s in self.sections:
            if s.name == name:
                return s
        return None
