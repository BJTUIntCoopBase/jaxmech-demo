"""jaxmech.materials - Constitutive model implementations."""

from importlib import import_module

from jaxmech.materials.base import ConstitutiveContext, ConstitutiveModel
from jaxmech.materials.definition import (
    LinearMaterialProperties,
    MaterialDef,
    resolve_linear_material_properties,
)

# Plastic definition helpers are optional (private modules).
try:
    from jaxmech.materials.definition import (
        J2PerfectPlasticProperties,
        resolve_j2_perfect_plastic_properties,
    )
    _HAS_PLASTIC_DEF = True
except ImportError:
    _HAS_PLASTIC_DEF = False

__all__ = [
    "ConstitutiveContext",
    "ConstitutiveModel",
    "IsotropicElastic",
    "LinearMaterialProperties",
    "MaterialDef",
    "d_matrix_isotropic_3d",
    "d_matrix_plane_stress",
    "d_matrix_shell_generalized",
    "get_d_matrix",
    "resolve_linear_material_properties",
]

if _HAS_PLASTIC_DEF:
    __all__ += [
        "J2PerfectPlastic",
        "J2PerfectPlasticProperties",
        "J2PlasticStateLayout",
        "J2PlasticStateView",
        "resolve_j2_perfect_plastic_properties",
    ]


_ELASTIC_EXPORTS = {
    "IsotropicElastic",
    "d_matrix_isotropic_3d",
    "d_matrix_plane_stress",
    "d_matrix_shell_generalized",
    "get_d_matrix",
}

_PLASTIC_EXPORTS = {
    "J2PerfectPlastic",
    "J2PlasticStateLayout",
    "J2PlasticStateView",
    "J2PerfectPlasticProperties",
    "resolve_j2_perfect_plastic_properties",
}


def __getattr__(name: str):
    """Lazily resolve JAX-dependent constitutive implementations."""
    if name in _ELASTIC_EXPORTS:
        module = import_module("jaxmech.materials.elastic")
        return getattr(module, name)
    if name in _PLASTIC_EXPORTS:
        try:
            module = import_module("jaxmech.materials.plastic")
            return getattr(module, name)
        except ImportError:
            pass
    # Also try definition-level plastic helpers via getattr fallback.
    if _HAS_PLASTIC_DEF and name in {"J2PerfectPlasticProperties", "resolve_j2_perfect_plastic_properties"}:
        from jaxmech.materials import definition as _def
        return getattr(_def, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
