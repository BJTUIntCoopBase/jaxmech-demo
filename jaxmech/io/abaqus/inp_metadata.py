"""Lightweight ABAQUS INP metadata parsing for Web and config flows.

This module extracts form-driving metadata without building a full JAX model.
It is intended for metadata-first workflows where Windows-side inspection must
not depend on meshio-backed solid parsing or solver imports.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jaxmech.io.abaqus.inp import _detect_element_family
from jaxmech.io.abaqus.solid_inp import _read_inp_text, parse_materials, parse_static_steps

try:
    from jaxmech.io.abaqus.shell_inp import parse_shell_inp
    _HAS_SHELL_INP = True
except ImportError:
    _HAS_SHELL_INP = False

log = logging.getLogger(__name__)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan"}:
            return None
        number = float(text)
        if math.isnan(number):
            return None
        return number
    except Exception:
        return None


def _default_increment_count(initial_increment: float | None, total_time: float | None) -> int:
    if initial_increment is None or total_time is None:
        return 1
    if initial_increment <= 0.0 or total_time <= 0.0:
        return 1
    steps = np.arange(
        initial_increment,
        total_time + 0.5 * initial_increment,
        initial_increment,
        dtype=np.float64,
    )
    if steps.size == 0 or steps[-1] < total_time - 1.0e-12:
        steps = np.append(steps, total_time)
    return int(steps.size)


def _build_step_info(step_meta: dict[str, Any], *, default_name: str) -> dict[str, Any]:
    initial_increment = _to_optional_float(step_meta.get("initial_increment"))
    total_time = _to_optional_float(step_meta.get("total_time"))
    return {
        "name": str(step_meta.get("step_name") or step_meta.get("name") or default_name),
        "procedure": str(step_meta.get("procedure", "") or ""),
        "initial_increment": initial_increment,
        "total_time": total_time,
        "min_increment": _to_optional_float(step_meta.get("min_increment")),
        "max_increment": _to_optional_float(step_meta.get("max_increment")),
        "nlgeom": bool(step_meta.get("nlgeom", False)),
        "default_n_increments": _default_increment_count(initial_increment, total_time),
    }


def _error_metadata(raw_path: str | Path, message: str) -> dict[str, Any]:
    path = Path(raw_path)
    return {
        "path": str(raw_path),
        "name": path.name,
        "error": message,
        "supported": False,
    }


def _parse_shell_metadata(inp_path: Path) -> dict[str, Any]:
    if not _HAS_SHELL_INP:
        return _error_metadata(inp_path, "Shell parsing is not available in this demo.")
    shell_data = parse_shell_inp(str(inp_path))
    materials = [
        {
            "index": 0,
            "name": str(shell_data.material_name),
            "elastic": {
                "E": float(shell_data.E),
                "nu": float(shell_data.nu),
                "thickness": float(shell_data.thickness),
                "type": "isotropic",
            },
        }
    ]
    return {
        "parser_scope": "shell_inp",
        "material_count": 1,
        "materials": materials,
        "has_plastic": False,
        "supports_elastic": True,
        "supports_elastoplastic": False,
        "suggested_material_model": "linear_elastic",
        "step": _build_step_info({}, default_name=inp_path.stem),
        "supported": True,
        "support_note": "shell 当前仅开放增量弹性。",
    }


def _parse_solid_metadata(inp_path: Path) -> dict[str, Any]:
    inp_text = _read_inp_text(str(inp_path))
    parsed_materials = parse_materials(inp_text)
    static_steps = parse_static_steps(inp_text)

    materials: list[dict[str, Any]] = []
    for idx, (mat_name, payload) in enumerate(sorted(parsed_materials.items())):
        material_info: dict[str, Any] = {
            "index": idx,
            "name": str(mat_name),
            "elastic": {
                "E": _to_optional_float(payload.get("E")),
                "nu": _to_optional_float(payload.get("nu")),
                "type": "isotropic",
            },
        }
        if "plastic_table" in payload:
            plastic_table = np.asarray(payload.get("plastic_table", []), dtype=np.float64)
            material_info["plastic"] = {
                "yield_stress": _to_optional_float(payload.get("yield_stress")),
                "hardening": str(payload.get("hardening", "") or ""),
                "table_rows": int(plastic_table.shape[0]) if plastic_table.ndim >= 2 else int(bool(plastic_table.size)),
            }
        materials.append(material_info)

    has_plastic = any(bool(material.get("plastic")) for material in materials)
    supports_elastoplastic = has_plastic and len(materials) == 1
    support_note = ""
    if len(materials) > 1:
        support_note = "当前 solid 增量分析仍只支持单材料 INP。"

    parser_scope = "solid_static_inp" if parsed_materials or static_steps else "elastic_inp_only"
    step_meta = dict(static_steps[0]) if static_steps else {}
    return {
        "parser_scope": parser_scope,
        "material_count": len(materials),
        "materials": materials,
        "has_plastic": has_plastic,
        "supports_elastic": True,
        "supports_elastoplastic": supports_elastoplastic,
        "suggested_material_model": "j2_perfect_plastic" if supports_elastoplastic else "linear_elastic",
        "step": _build_step_info(step_meta, default_name=inp_path.stem),
        "supported": len(materials) == 1,
        "support_note": support_note,
    }


def parse_inp_metadata(inp_path: str | Path) -> dict[str, Any]:
    """Return lightweight, JSON-serializable metadata for one INP file."""
    raw_path = str(inp_path)
    path = Path(raw_path)
    if not path.is_file():
        return _error_metadata(raw_path, f"INP file not found: {raw_path}")

    try:
        family = _detect_element_family(str(path)).strip().lower()
        payload = _parse_shell_metadata(path) if family == "shell" else _parse_solid_metadata(path)
        return {
            "path": raw_path,
            "name": path.name,
            "family": family,
            **payload,
        }
    except Exception as exc:
        log.exception("Failed to parse INP metadata: %s", raw_path)
        return _error_metadata(raw_path, str(exc))


def parse_inp_metadata_batch(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Return metadata for multiple INP files, preserving input order."""
    return [parse_inp_metadata(path) for path in paths]


__all__ = [
    "parse_inp_metadata",
    "parse_inp_metadata_batch",
]