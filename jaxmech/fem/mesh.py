"""
jaxmech.fem.mesh - Mesh and element-block data structures.

These are finite-element discretization objects and therefore belong to the
FEM layer rather than the higher-level model contract layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ElementBlock:
    """A group of elements sharing the same cell_type and ele_type."""

    block_id: int
    cell_type: str
    ele_type: str
    connectivity: np.ndarray
    element_ids: Optional[np.ndarray] = None
    elset_name: Optional[str] = None

    @property
    def n_elements(self) -> int:
        return self.connectivity.shape[0]

    @property
    def nodes_per_elem(self) -> int:
        return self.connectivity.shape[1]


@dataclass
class Mesh:
    """Finite element mesh: nodes + element blocks."""

    nodes: np.ndarray
    blocks: list[ElementBlock] = field(default_factory=list)
    node_ids: Optional[np.ndarray] = None
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    element_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def n_elements_total(self) -> int:
        return sum(block.n_elements for block in self.blocks)

    def get_block_by_ele_type(self, ele_type: str) -> list[ElementBlock]:
        return [block for block in self.blocks if block.ele_type == ele_type]

    def get_block_by_cell_type(self, cell_type: str) -> list[ElementBlock]:
        return [block for block in self.blocks if block.cell_type == cell_type]
