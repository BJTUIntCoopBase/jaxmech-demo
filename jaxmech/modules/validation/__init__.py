"""jaxmech.modules.validation - Validation analysis modules."""

from jaxmech.modules.validation.elastic import (
    ValidationReport,
    run_elastic_validation,
    validate_elastic_odb,
)

__all__ = [
    "ValidationReport",
    "run_elastic_validation",
    "validate_elastic_odb",
]

# Nonlinear validation is optional (private modules).
try:
    from jaxmech.modules.validation.nonlinear import (
        compare_direct_cyclic_mat_with_abaqus_mat,
        run_nonlinear_validation,
        validate_nonlinear_mat_with_abaqus_odb,
    )
    __all__ += [
        "compare_direct_cyclic_mat_with_abaqus_mat",
        "run_nonlinear_validation",
        "validate_nonlinear_mat_with_abaqus_odb",
    ]
except ImportError:
    pass
