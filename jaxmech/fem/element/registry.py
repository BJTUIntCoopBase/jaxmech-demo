"""
jaxmech.fem.element.registry — Element type registry mapping.
"""

from typing import Dict, Type, Optional

from jaxmech.fem.element.core import ElementKernel
from jaxmech.fem.element.solid import (
    C3D8Kernel, C3D8RKernel, C3D6Kernel, C3D4Kernel
)

# Mapping from Abaqus ele_type string to ElementKernel class
_KERNEL_REGISTRY: Dict[str, Type[ElementKernel]] = {
    "C3D8": C3D8Kernel,
    "C3D8R": C3D8RKernel,
    "C3D6": C3D6Kernel,
    "C3D4": C3D4Kernel,
}

# Default variants for elements that support them
_DEFAULT_VARIANTS: Dict[str, str] = {
    "C3D8": "bbar",
}


def get_kernel(ele_type: str, *, variant: Optional[str] = None) -> ElementKernel:
    """Retrieve an instantiated ElementKernel for the given ABAQUS element type.

    Parameters
    ----------
    ele_type : str
        ABAQUS element type string (e.g. 'C3D8', 'CPS4').
    variant : str, optional
        Element variant (e.g. 'bbar', 'standard').  If None, uses the
        registered default (B-bar for C3D8, standard otherwise).
    """
    ele_type_upper = ele_type.upper()
    if ele_type_upper not in _KERNEL_REGISTRY:
        raise ValueError(
            f"Element type '{ele_type_upper}' is not registered. "
            f"Available: {list(_KERNEL_REGISTRY.keys())}"
        )

    cls = _KERNEL_REGISTRY[ele_type_upper]

    # Pass variant to kernels that accept it
    if variant is not None:
        try:
            return cls(variant=variant)
        except TypeError:
            return cls()

    default_variant = _DEFAULT_VARIANTS.get(ele_type_upper)
    if default_variant is not None:
        try:
            return cls(variant=default_variant)
        except TypeError:
            return cls()

    return cls()
