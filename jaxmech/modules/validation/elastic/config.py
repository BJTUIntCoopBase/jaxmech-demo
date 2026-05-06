"""Configuration parsing for elastic validation workflows."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import scipy.io as sio

from jaxmech.env.workflow_files import find_unique_workflow_dir, resolve_workflow_target
from jaxmech.env.model_paths import win_to_wsl_path


DEFAULT_CONFIG = {
    "mat_files": [],
    "odb_files": [],
    "copy_mat": True,
    "windows_abaqus_cmd": "",
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


def _parse_path_list(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item).strip()]
    raise ValueError("Expected one path string or a list/tuple of paths.")


def _normalize_cfg_path(path_str: str | Path) -> Path:
    return win_to_wsl_path(str(path_str)).resolve()


def _infer_family_from_mat(mat_path: str | Path) -> str:
    raw = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    if "solid_elem_types" in raw:
        return "solid"
    if any(key in raw for key in ("shell_u_nodal", "shell_layer_stress", "C_gen", "shell_elem_thickness")):
        return "shell"
    if "abaqus_elem_types" in raw:
        flat = np.asarray(raw["abaqus_elem_types"], dtype=object).reshape(-1)
        if any("S" in str(item).upper() and not str(item).upper().startswith(("CPS", "CPE")) for item in flat):
            return "shell"
    return "solid"


def parse_validation_config(target: str | Path) -> dict:
    resolved = resolve_workflow_target(target, "validation")
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
            if key in {"mat_files", "odb_files"}:
                config[key] = _parse_path_list(value)
            elif key == "copy_mat":
                config[key] = bool(ast.literal_eval(value))
            elif key == "windows_abaqus_cmd":
                config[key] = value
            else:
                config[key] = value

    if config["mat_files"]:
        mat_files = [_normalize_cfg_path(path) for path in config["mat_files"]]
    else:
        inc_dir = find_unique_workflow_dir(model_dir, "inc_analysis")
        if inc_dir is None:
            raise FileNotFoundError(
                f"No validation mat_files provided and no sibling inc_analysis directory found under {model_dir}."
            )
        mat_files = sorted(Path(inc_dir).glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError("No source .mat files found for validation.")

    if config["odb_files"]:
        odb_files = [_normalize_cfg_path(path) for path in config["odb_files"]]
    else:
        odb_files = [abaqus_dir / f"{mat_path.stem}.odb" for mat_path in mat_files]

    if len(mat_files) != len(odb_files):
        raise ValueError("mat_files and odb_files must have the same length.")

    config["family"] = _infer_family_from_mat(mat_files[0])
    config["mat_files"] = mat_files
    config["odb_files"] = odb_files
    return config
