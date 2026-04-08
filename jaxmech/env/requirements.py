"""Workflow-level machine configuration checks."""

from __future__ import annotations

from typing import Iterable

from jaxmech.env.env import ENV_CFG, load_machine_config


def _blank(value: object) -> bool:
    return str(value).strip() == ""


def missing_machine_keys(keys: Iterable[str]) -> list[str]:
    config = load_machine_config()
    return [key for key in keys if _blank(config.get(key, ""))]


def require_machine_keys(keys: Iterable[str], *, context: str) -> None:
    missing = missing_machine_keys(keys)
    if not missing:
        return
    raise RuntimeError(
        f"{context} requires machine config keys in {ENV_CFG}: "
        + ", ".join(missing)
    )


def require_shakedown_keys(config: dict, keys: Iterable[str], *, context: str) -> None:
    missing: list[str] = []
    for key in keys:
        value = config.get(key, "")
        if _blank(value):
            missing.append(key)
    if not missing:
        return
    raise RuntimeError(
        f"{context} requires config values in the shakedown .cfg: "
        + ", ".join(missing)
    )


def validate_shakedown_machine_config(config: dict, *, step: str) -> None:
    solver = str(config.get("solver", ""))
    if "+Gurobi" not in solver:
        return

    backend = str(config.get("gurobi_backend", "python")).strip().lower()
    # windows_gurobi_root is always required; exe depends on backend
    required = ["windows_gurobi_root"]
    if backend == "python":
        required.append("windows_python_exe")
    elif backend == "matlab":
        required.append("windows_matlab_exe")
    else:
        raise RuntimeError(f"Unsupported shakedown gurobi_backend: {backend!r}")
    # windows_gurobi_user is optional (only for Academic License)
    if step in {"all", "prepare"}:
        require_shakedown_keys(config, required, context=f"shakedown solver {solver!r}")
