"""
jaxmech.model.boundary - Boundary condition definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np


@dataclass
class BoundaryCondition:
    """A Dirichlet (displacement) boundary condition.

    Attributes:
        node_set:   Name of the node set, or array of node indices.
        dof_first:  First constrained DOF (1-based, ABAQUS convention).
        dof_last:   Last constrained DOF (1-based, ABAQUS convention).
                    If equal to dof_first, only one DOF is constrained.
        value:      Prescribed displacement value (default 0.0 = homogeneous).
    """
    node_set: str | np.ndarray
    dof_first: int
    dof_last: int
    value: float = 0.0

    @property
    def dof_range_0based(self) -> range:
        """Return 0-based DOF range."""
        return range(self.dof_first - 1, self.dof_last)
