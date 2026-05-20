"""
jaxmech.materials.definition - Material data definitions and resolution helpers.

This file owns the data side of materials. Constitutive computations remain in
``jaxmech.materials`` proper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from jaxmech.model.model import Model


@dataclass
class MaterialDef:
    """Material property definition (data container, not constitutive law)."""

    name: str
    elastic: Dict[str, Any] = field(default_factory=dict)
    density: Optional[float] = None
    plastic: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def E(self) -> float:
        return self.elastic["E"]

    @property
    def nu(self) -> float:
        return self.elastic["nu"]


@dataclass(frozen=True)
class LinearMaterialProperties:
    """Resolved material properties for a single linear analysis step."""

    E: float
    nu: float
    thickness: float = 1.0
    model_name: str = "linear_elastic"
    source_material: str = ""

    def to_dict(self, *, use_b_ext: int) -> Dict[str, Any]:
        return {
            "E": self.E,
            "nu": self.nu,
            "thickness": self.thickness,
            "use_b_ext": use_b_ext,
        }


@dataclass(frozen=True)
class J2PerfectPlasticProperties:
    """Resolved material properties for small-strain J2 ideal plasticity."""

    E: float
    nu: float
    yield_stress: float
    thickness: float = 1.0
    model_name: str = "j2_perfect_plastic"
    hardening: str = "perfect"
    source_material: str = ""
    density: Optional[float] = None

    def to_dict(self, *, use_b_ext: int) -> Dict[str, Any]:
        return {
            "E": self.E,
            "nu": self.nu,
            "yield_stress": self.yield_stress,
            "thickness": self.thickness,
            "use_b_ext": use_b_ext,
            "hardening": self.hardening,
        }


def resolve_linear_material_properties(
    model: "Model",
    *,
    E_override: Optional[float] = None,
    nu_override: Optional[float] = None,
) -> LinearMaterialProperties:
    """Resolve the active material payload for linear analysis from ``Model``."""
    if not model.materials:
        raise ValueError("Model has no materials configured.")

    material = model.materials[0]
    family = model.metadata.get("family", "solid")

    E = E_override if E_override is not None else material.elastic.get("E", 210000.0)
    nu = nu_override if nu_override is not None else material.elastic.get("nu", 0.3)

    thickness = 1.0
    if family == "shell" and model.sections and hasattr(model.sections[0], "thickness"):
        thickness = float(model.sections[0].thickness)

    return LinearMaterialProperties(
        E=float(E),
        nu=float(nu),
        thickness=float(thickness),
        model_name=str(material.elastic.get("type", "linear_elastic")),
        source_material=material.name,
    )


def resolve_j2_perfect_plastic_properties(
    model: "Model",
    *,
    E_override: Optional[float] = None,
    nu_override: Optional[float] = None,
    yield_stress_override: Optional[float] = None,
) -> J2PerfectPlasticProperties:
    """Resolve the active material payload for small-strain J2 ideal plasticity."""
    if not model.materials:
        raise ValueError("Model has no materials configured.")

    material = model.materials[0]
    family = model.metadata.get("family", "solid")

    E = E_override if E_override is not None else material.elastic.get("E", 210000.0)
    nu = nu_override if nu_override is not None else material.elastic.get("nu", 0.3)

    thickness = 1.0
    if family == "shell" and model.sections and hasattr(model.sections[0], "thickness"):
        thickness = float(model.sections[0].thickness)

    plastic = material.plastic or {}
    yield_stress = yield_stress_override
    if yield_stress is None:
        for key in ("yield_stress", "sigma_y", "yield"):
            if key in plastic:
                yield_stress = float(plastic[key])
                break

    if yield_stress is None and "table" in plastic:
        table = np.asarray(plastic["table"], dtype=np.float64).reshape(-1)
        if table.size:
            yield_stress = float(table[0])

    if yield_stress is None:
        raise ValueError(
            "The active material does not define a yield stress for J2 perfect plasticity."
        )

    return J2PerfectPlasticProperties(
        E=float(E),
        nu=float(nu),
        yield_stress=float(yield_stress),
        thickness=float(thickness),
        source_material=material.name,
        density=material.density,
    )
