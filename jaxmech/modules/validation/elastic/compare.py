"""Elastic ODB comparison for solid and shell analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ValidationReport:
    """Structured result of one elastic ODB validation case."""

    errors: dict[str, float] = field(default_factory=dict)
    family: str = "solid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"ODB Validation (family={self.family})"
        ]
        for key, val in sorted(self.errors.items()):
            lines.append(f"  {key:8s}: {val:.6e}")
        if "error" in self.metadata:
            lines.append(f"  error: {self.metadata['error']}")
        return "\n".join(lines)


def validate_elastic_odb(
    config_path: str | Path,
    *,
    force: bool = False,
    abaqus_mat_dir: str | Path | None = None,
    skip_extraction: bool = True,
    family: str = "solid",
) -> list[ValidationReport]:
    """Validate elastic JAX results against ABAQUS ODB-derived data."""
    family_key = str(family).strip().lower()
    if family_key == "shell":
        return _validate_shell(
            config_path,
            force=force,
            abaqus_mat_dir=abaqus_mat_dir,
            skip_extraction=skip_extraction,
        )
    if family_key == "solid":
        return _validate_solid(
            config_path,
            force=force,
            abaqus_mat_dir=abaqus_mat_dir,
            skip_extraction=skip_extraction,
        )
    raise ValueError(f"Unsupported elastic validation family: {family!r}")


def _validate_solid(
    config_path: str | Path,
    *,
    force: bool,
    abaqus_mat_dir: str | Path | None,
    skip_extraction: bool,
) -> list[ValidationReport]:
    """Dispatch to the current solid elastic validation implementation."""
    from jaxmech.modules.validation.elastic.solid_odb import run_odb_validation

    raw_results = run_odb_validation(
        config_path=str(config_path),
        force=force,
        abaqus_mat_dir=str(abaqus_mat_dir) if abaqus_mat_dir else None,
        skip_extraction=skip_extraction,
    )

    reports: list[ValidationReport] = []
    for raw in raw_results:
        errors = raw.get("errors", {})
        err_dict = {
            key: float(val)
            for key, val in errors.items()
            if isinstance(val, (int, float))
        } if isinstance(errors, dict) else {}
        reports.append(
            ValidationReport(
                errors=err_dict,
                family="solid",
                metadata={
                    "case_name": raw.get("case_name", ""),
                    "skipped": raw.get("skipped", False),
                    "abaqus_mat_path": raw.get("abaqus_mat_path", ""),
                    "validated_odb_name": raw.get("validated_ODBName", ""),
                },
            )
        )
    return reports


def _validate_shell(
    config_path: str | Path,
    *,
    force: bool,
    abaqus_mat_dir: str | Path | None,
    skip_extraction: bool,
) -> list[ValidationReport]:
    """Dispatch to the current shell elastic validation implementation."""
    from jaxmech.modules.validation.elastic.shell_odb import run_shell_odb_validation

    raw_results = run_shell_odb_validation(
        config_path=str(config_path),
        force=force,
        abaqus_mat_dir=str(abaqus_mat_dir) if abaqus_mat_dir else None,
        skip_extraction=skip_extraction,
    )

    reports: list[ValidationReport] = []
    for raw in raw_results:
        errors = raw.get("errors", {})
        err_dict = {
            key: float(val)
            for key, val in errors.items()
            if isinstance(val, (int, float))
        } if isinstance(errors, dict) else {}
        reports.append(
            ValidationReport(
                errors=err_dict,
                family="shell",
                metadata={
                    "case_name": raw.get("case_name", ""),
                    "skipped": raw.get("skipped", False),
                    "abaqus_mat_path": raw.get("validated_ODBName", ""),
                },
            )
        )
    return reports


__all__ = [
    "ValidationReport",
    "validate_elastic_odb",
]
