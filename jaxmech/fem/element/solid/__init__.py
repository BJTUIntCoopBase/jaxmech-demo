"""
jaxmech.fem.element.solid — Solid continuum elements (2D and 3D).
"""

from .c3d8 import C3D8Kernel
from .c3d8r import C3D8RKernel
from .c3d6 import C3D6Kernel
from .c3d4 import C3D4Kernel

__all__ = [
    "C3D8Kernel",
    "C3D8RKernel",
    "C3D6Kernel",
    "C3D4Kernel",
]
