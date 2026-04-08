"""Unified shakedown analysis module (demo: solid C formulation only)."""

from __future__ import annotations

from pathlib import Path

from jaxmech.modules.shakedown.result import ShakedownResult


SOLID_SOLVERS = {"VersionC"}

__all__ = [
    "ShakedownResult",
    "run_shakedown",
]


def run_shakedown(
    config_path: str | Path,
    *,
    step: str = "all",
) -> ShakedownResult:
    """Run a cfg-driven shakedown analysis."""
    from jaxmech.env.requirements import validate_shakedown_machine_config
    from jaxmech.modules.shakedown.config import parse_shakedown_config

    config = parse_shakedown_config(str(config_path))
    validate_shakedown_machine_config(config, step=step)
    solver_key = str(config["solver"])

    if solver_key in SOLID_SOLVERS:
        return _run_solid(config_path, config, step=step)
    raise ValueError(
        f"Unknown solver: {solver_key!r}. "
        f"This demo only supports solid C formulation with CVXPY. "
        f"Supported: {sorted(SOLID_SOLVERS)}"
    )


def _run_solid(
    config_path: str | Path,
    config: dict,
    step: str,
) -> ShakedownResult:
    """Dispatch to the native solid shakedown pipeline."""
    from jaxmech.modules.shakedown.native_solid import run_native_solid

    return run_native_solid(config_path, config, step=step)
