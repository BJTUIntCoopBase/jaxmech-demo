"""Configuration parsing for the demo solid CVXPY shakedown workflow."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from jaxmech.env.env import load_machine_config
from jaxmech.env.model_paths import win_to_wsl_path


_MACHINE_CONFIG = load_machine_config()

DEFAULT_CONFIG = {
    "mat_files": [],
    "formulation": "C",
    "solver_backend": "cvxpy",
    "solver": "VersionC",
    "family": "solid",
    "wsl_python": _MACHINE_CONFIG.get("wsl_python", ""),
    "load_cases": [],
    "R_ratios": [0.0],
    "use_angles": True,
    "theta_deg": [45.0],
    "phi_deg": [35.26438968],
    "load_factor_set": None,
    "yield_values": {"MATERIAL-1": 280.0},
    "num_vert": 2,
}


def parse_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _normalize_host_path(path_text: str) -> str:
    if os.name == "nt":
        return str(Path(path_text))
    return str(win_to_wsl_path(path_text))


def parse_mat_files(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, str):
        out = [parsed]
    elif isinstance(parsed, (list, tuple)):
        out = [str(item) for item in parsed if str(item).strip()]
    else:
        raise ValueError("mat_files must be a string path or a list/tuple of paths.")
    if not 1 <= len(out) <= 3:
        raise ValueError(f"mat_files must contain 1 to 3 items, got {len(out)}.")
    return [_normalize_host_path(item) for item in out]


def parse_angle_array(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [float(item) for item in parsed]
    raise ValueError(f"Cannot parse numeric array: {value}")


def parse_load_factor_set(value: str, n_cases: int) -> list[list[float]]:
    rows = np.asarray(ast.literal_eval(value), dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < n_cases:
        raise ValueError(f"load_factor_set must have at least {n_cases} columns, got {rows.shape}.")
    return rows[:, :n_cases].tolist()


def parse_yield_values(value: str):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, dict):
        return {str(key).upper(): float(val) for key, val in parsed.items()}
    raise ValueError("yield_values must be a scalar or a material-name dictionary.")


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


def _metadata_to_dict(raw_meta) -> dict[str, object]:
    if raw_meta is None:
        return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    if hasattr(raw_meta, "_fieldnames"):
        return {name: getattr(raw_meta, name) for name in raw_meta._fieldnames}
    if isinstance(raw_meta, np.ndarray) and getattr(raw_meta.dtype, "names", None):
        rec = raw_meta.reshape(-1)[0]
        return {name: rec[name] for name in raw_meta.dtype.names}
    return {}


def _infer_family_from_mat(mat_file: str) -> str:
    raw = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False, variable_names=["metadata", "InpData"])
    metadata = _metadata_to_dict(raw.get("metadata"))
    family = str(np.asarray(metadata.get("family", "")).reshape(-1)[0]).strip().lower() if metadata else ""
    if family and family != "solid":
        raise ValueError(f"The demo shakedown workflow only supports solid MAT files, got {family!r}.")
    return "solid"


def parse_shakedown_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        raise ValueError("Shakedown config must define mat_files.")
    config = dict(DEFAULT_CONFIG)
    config_path = Path(config_path).resolve()
    config["config_path"] = str(config_path)
    config["workflow_dir"] = str(config_path.parent)

    for key, value in _collect_cfg_items(config_path):
        if key == "mat_files":
            config[key] = parse_mat_files(value)
        elif key == "formulation":
            config[key] = value.upper()
        elif key in {"solver", "solver_backend"}:
            config["solver_backend"] = value.lower()
        elif key == "wsl_python":
            config[key] = value
        elif key == "R_ratios":
            config[key] = parse_angle_array(value)
        elif key == "use_angles":
            config[key] = parse_bool(value)
        elif key == "theta_deg":
            config[key] = parse_angle_array(value)
        elif key == "phi_deg":
            config[key] = parse_angle_array(value)
        elif key == "load_factor_set":
            config["_lfs_raw"] = value
        elif key == "yield_values":
            config[key] = parse_yield_values(value)
        elif key == "num_vert":
            config[key] = int(value)

    if not config["mat_files"]:
        raise ValueError("Shakedown config must define mat_files.")
    if config["formulation"] != "C":
        raise ValueError("The demo shakedown workflow only supports formulation = C.")
    if config["solver_backend"] != "cvxpy":
        raise ValueError("The demo shakedown workflow only supports solver = cvxpy.")

    config["family"] = _infer_family_from_mat(config["mat_files"][0])
    config["solver"] = "VersionC"
    config["load_cases"] = [Path(path).stem for path in config["mat_files"]]

    n_cases = len(config["mat_files"])
    ratios = list(config["R_ratios"])
    if len(ratios) < n_cases:
        config["R_ratios"] = ratios + [0.0] * (n_cases - len(ratios))
    elif len(ratios) > n_cases:
        raise ValueError(f"R_ratios has {len(ratios)} values but {n_cases} MAT files are specified.")

    if "_lfs_raw" in config:
        config["load_factor_set"] = parse_load_factor_set(config.pop("_lfs_raw"), n_cases)
    else:
        config.pop("_lfs_raw", None)

    if int(config["num_vert"]) not in {1, 2, 4, 8}:
        raise ValueError("num_vert must be one of 1, 2, 4, or 8.")
    if int(config["num_vert"]) not in {1, 2} and int(config["num_vert"]) != 2 ** n_cases:
        raise ValueError(f"For {n_cases} load cases, num_vert must be 1, 2, or {2 ** n_cases}.")
    if not config["use_angles"] and config["load_factor_set"] is None:
        raise ValueError("When use_angles = no, load_factor_set must be provided.")
    if config["use_angles"]:
        config["load_factor_set"] = None

    if not str(config.get("wsl_python", "")).strip() and _MACHINE_CONFIG.get("wsl_python"):
        config["wsl_python"] = _MACHINE_CONFIG["wsl_python"]
    return config
