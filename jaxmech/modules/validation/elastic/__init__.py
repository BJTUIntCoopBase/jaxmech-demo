"""Elastic validation workflows for solid and shell analyses."""

from jaxmech.modules.validation.elastic.compare import (
    ValidationReport,
    validate_elastic_odb,
)


def run_elastic_validation(*args, **kwargs):
    from jaxmech.modules.validation.elastic.run import run_elastic_validation as _run

    return _run(*args, **kwargs)

__all__ = [
    "ValidationReport",
    "run_elastic_validation",
    "validate_elastic_odb",
]
