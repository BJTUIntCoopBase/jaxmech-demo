"""Configuration parsing for MAT-first elastic analysis workflows."""

from __future__ import annotations

import ast
from pathlib import Path

from jaxmech.env.workflow_files import resolve_workflow_target


DEFAULT_CONFIG = {
    "inp_files": [],
    "use_b_ext": 0,
    "gauss_order": 2,
    "material_model": "auto",
    "n_increments": 1,
    "max_iterations": 25,
    "convergence_tol": 1.0e-8,
    "E_override": None,
    "nu_override": None,
    "yield_stress_override": None,
}


def _clean_cfg_lines(config_path: str | Path) -> list[str]:
    lines: list[str] = []
    for raw_line in Path(config_path).read_text(encoding="utf-8").splitlines():
        data = raw_line.split("#", 1)[0].strip()
        if data:
            lines.append(data)
    return lines


def _collect_cfg_items(config_path: str | Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    lines = _clean_cfg_lines(config_path)
    i = 0
    while i < len(lines):
        line = lines[i]
        if "=" not in line:
            i += 1
            continue
        key, value = [item.strip() for item in line.split("=", 1)]
        balance = sum(value.count(ch) for ch in "([{") - sum(value.count(ch) for ch in ")]}")
        i += 1
        while balance > 0 and i < len(lines):
            value += " " + lines[i]
            balance += sum(lines[i].count(ch) for ch in "([{") - sum(lines[i].count(ch) for ch in ")]}")
            i += 1
        items.append((key, value))
    return items


def _parse_inp_files(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item).strip()]
    raise ValueError("inp_files must be a string path or a list/tuple of paths.")


def _parse_optional_float(value: str) -> float | None:
    parsed = ast.literal_eval(value)
    if parsed is None:
        return None
    if isinstance(parsed, str) and not parsed.strip():
        return None
    return float(parsed)


def _parse_string_value(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = value
    return str(parsed).strip()


def parse_inc_analysis_config(target: str | Path) -> dict:
    resolved = resolve_workflow_target(target, "inc_analysis")
    workflow_dir = Path(resolved["workflow_dir"]).resolve()
    model_dir = Path(resolved["model_dir"]).resolve()
    abaqus_dir = model_dir / "abaqus"

    config = dict(DEFAULT_CONFIG)
    config["workflow_dir"] = workflow_dir
    config["model_dir"] = model_dir
    config["config_path"] = resolved["config_path"]

    config_path = resolved["config_path"]
    if config_path is not None:
        for key, value in _collect_cfg_items(config_path):
            if key == "inp_files":
                config[key] = _parse_inp_files(value)
            elif key in {"use_b_ext", "gauss_order", "n_increments", "max_iterations"}:
                config[key] = int(ast.literal_eval(value))
            elif key in {"convergence_tol", "E_override", "nu_override", "yield_stress_override"}:
                config[key] = _parse_optional_float(value)
            elif key == "material_model":
                config[key] = _parse_string_value(value)
            else:
                config[key] = value

    if config["inp_files"]:
        inp_files = [Path(path).resolve() for path in config["inp_files"]]
    else:
        inp_files = sorted(abaqus_dir.glob("*.inp"))
    if not inp_files:
        raise FileNotFoundError(
            f"No .inp files found for inc_analysis under {abaqus_dir}. "
            "Provide inp_files in the workflow cfg or place INP files in model/abaqus."
        )

    config["inp_files"] = inp_files
    return config
