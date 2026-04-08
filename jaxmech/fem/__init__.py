"""jaxmech.fem - Finite element core (mesh, sections, elements, quadrature, assembly, solver)."""

from jaxmech.fem.mesh import ElementBlock, Mesh
from jaxmech.fem.section import Section, ShellSection, SolidSection

__all__ = [
    "ElementBlock",
    "Mesh",
    "Section",
    "ShellSection",
    "SolidSection",
]
