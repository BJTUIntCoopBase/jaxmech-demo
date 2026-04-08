"""
jaxmech.fem.section - Section assignment definitions.

Sections are part of the finite-element discretization because they bind
material and geometric section data to element regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Section:
    name: str
    material_name: str
    elset_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolidSection(Section):
    thickness: float = 1.0


@dataclass
class ShellSection(Section):
    thickness: float = 1.0
    num_int_pts: int = 5
    offset: float = 0.0
