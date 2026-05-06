"""Lightweight solid INP metadata parsing for Web forms."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jaxmech.io.abaqus.inp import _detect_element_family
from jaxmech.io.abaqus.solid_inp import _read_inp_text, parse_materials, parse_static_steps


log = logging.getLogger(__name__)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan"}:
            return None
        number = float(text)
        return None if math.isnan(number) else number
    except Exception:
        return None


def _default_increment_count(initial_increment: float | None, total_time: float | None) -> int:
    if initial_increment is None or total_time is None or initial_increment <= 0.0 or total_time <= 0.0:
        return 1
    steps = np.arange(initial_increment, total_time + 0.5 * initial_increment, initial_increment, dtype=np.float64)
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
    return {"path": str(raw_path), "name": path.name, "error": message, "supported": False}


def _parse_solid_metadata(inp_path: Path) -> dict[str, Any]:
    inp_text = _read_inp_text(str(inp_path))
    parsed_materials = parse_materials(inp_text)
    static_steps = parse_static_steps(inp_text)

    materials: list[dict[str, Any]] = []
    for idx, (mat_name, payload) in enumerate(sorted(parsed_materials.items())):
        materials.append(
            {
                "index": idx,
                "name": str(mat_name),
                "elastic": {
                    "E": _to_optional_float(payload.get("E")),
                    "nu": _to_optional_float(payload.get("nu")),
                    "type": "isotropic",
                },
            }
        )

    step_meta = dict(static_steps[0]) if static_steps else {}
    supported = len(materials) == 1
    return {
        "parser_scope": "solid_elastic_inp",
        "material_count": len(materials),
        "materials": materials,
        "has_plastic": False,
        "supports_elastic": True,
        "supports_elastoplastic": False,
        "suggested_material_model": "linear_elastic",
        "step": _build_step_info(step_meta, default_name=inp_path.stem),
        "supported": supported,
        "support_note": "" if supported else "Demo inc_analysis supports one solid elastic material.",
    }


def parse_inp_metadata(inp_path: str | Path) -> dict[str, Any]:
    raw_path = str(inp_path)
    path = Path(raw_path)
    if not path.is_file():
        return _error_metadata(raw_path, f"INP file not found: {raw_path}")
    try:
        family = _detect_element_family(str(path)).strip().lower()
        if family != "solid":
            payload = _error_metadata(raw_path, "The demo supports solid element INP files only.")
            payload["family"] = family
            return payload
        return {"path": raw_path, "name": path.name, "family": "solid", **_parse_solid_metadata(path)}
    except Exception as exc:
        log.exception("Failed to parse INP metadata: %s", raw_path)
        return _error_metadata(raw_path, str(exc))


def parse_inp_metadata_batch(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [parse_inp_metadata(path) for path in paths]


__all__ = ["parse_inp_metadata", "parse_inp_metadata_batch"]
