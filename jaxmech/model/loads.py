"""
jaxmech.model.loads - Load case definitions.

A LoadCase collects all loads applied in a single analysis step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np


@dataclass
class ConcentratedLoad:
    """A concentrated (nodal) load.

    Attributes:
        node_set:  Name of the node set, or array of node indices.
        dof:       Degree of freedom index (0-based) or list of DOF indices.
        magnitude: Load magnitude(s). Scalar or array matching node_set size.
    """
    node_set: str | np.ndarray
    dof: int | List[int]
    magnitude: float | np.ndarray


@dataclass
class SurfaceLoad:
    """A distributed surface load (pressure, traction).

    Attributes:
        surface_name: ABAQUS surface name.
        load_type:    Load type string, e.g. "P" (pressure), "TRVEC" (traction).
        magnitude:    Load magnitude.
        direction:    Optional direction vector for traction loads.
    """
    surface_name: str
    load_type: str = "P"
    magnitude: float = 0.0
    direction: Optional[np.ndarray] = None


@dataclass
class LoadCase:
    """A single load case (analysis step).

    Attributes:
        name:              Load case name (e.g. "Load1Ela").
        concentrated:      List of concentrated loads.
        surface_loads:      List of surface loads.
        body_forces:       Body force vector (optional).
        metadata:          Free-form metadata.
    """
    name: str
    concentrated: List[ConcentratedLoad] = field(default_factory=list)
    surface_loads: List[SurfaceLoad] = field(default_factory=list)
    body_forces: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
