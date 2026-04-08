"""jaxmech.model - Unified data model for computational mechanics."""

from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult, Operators
from jaxmech.model.loads import LoadCase, SurfaceLoad, ConcentratedLoad
from jaxmech.model.boundary import BoundaryCondition

__all__ = [
    "Model",
    "AnalysisResult", "Operators",
    "LoadCase", "SurfaceLoad", "ConcentratedLoad",
    "BoundaryCondition",
]
