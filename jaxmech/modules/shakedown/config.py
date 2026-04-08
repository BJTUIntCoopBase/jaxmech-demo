"""Shakedown module configuration parsing."""

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
    "shell_yield": "Ilyushin",
    "ilyushin_c": 0.5,
    "solver_backend": "cvxpy",
    "solver": "",
    "family": "",
    "gurobi_backend": "python",
    "windows_gurobi_root": "",
    "windows_python_exe": "",
    "windows_matlab_exe": "",
    "windows_gurobi_user": "",
    "windows_gurobi_password": "",
    "gurobi_params": {"TimeLimit": 3600, "FeasibilityTol": 1e-6},
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
    val = value.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


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


def _normalize_host_path(path_text: str) -> str:
    if os.name == "nt":
        return str(Path(path_text))
    return str(win_to_wsl_path(path_text))


def parse_angle_array(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [float(x) for x in parsed]
    raise ValueError(f"Cannot parse angle array: {value}")


def parse_ratio_array(value: str) -> list[float]:
    """Parse R_ratios; accept 1-3 values (padding to n_cases is done later)."""
    arr = parse_angle_array(value)
    if not 1 <= len(arr) <= 3:
        raise ValueError(f"R_ratios must contain 1 to 3 values, got {len(arr)}.")
    return arr


def parse_load_factor_set(value: str, n_cases: int = 3) -> list[list[float]]:
    """Parse load_factor_set; number of columns must equal n_cases."""
    parsed = ast.literal_eval(value)
    rows = np.asarray(parsed, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != n_cases:
        raise ValueError(
            f"load_factor_set must have shape (n_vert, {n_cases}), got {rows.shape}."
        )
    return rows.tolist()


def parse_yield_values(value: str):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, dict):
        return {str(k).upper(): float(v) for k, v in parsed.items()}
    raise ValueError(
        "yield_values must be a scalar or a dict like "
        "{'MATERIAL-1':280, 'MATERIAL-2':350}."
    )


def parse_literal_dict(value: str):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, dict):
        return parsed
    raise ValueError(f"Expected a dict literal, got: {value}")


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
    raw = sio.loadmat(mat_file, squeeze_me=False, struct_as_record=False)
    metadata = _metadata_to_dict(raw.get("metadata"))
    meta_family = str(np.asarray(metadata.get("family", "")).reshape(-1)[0]).strip().lower() if metadata else ""
    if meta_family in {"solid", "shell"}:
        return meta_family
    if any(key in raw for key in ("shell_u_nodal", "shell_layer_stress", "C_gen")):
        return "shell"
    return "solid"


def _resolve_solver_key(config: dict) -> str:
    family = str(config["family"]).lower()
    formulation = str(config["formulation"]).upper()
    shell_yield = str(config["shell_yield"]).lower()
    backend = str(config["solver_backend"]).lower()

    if family == "solid":
        if formulation not in {"C", "MN"}:
            raise ValueError(f"Solid shakedown formulation must be C or MN, got {formulation!r}.")
        if backend == "cvxpy":
            return "VersionC" if formulation == "C" else "VersionMN"
        if backend == "gurobi":
            return "VersionC+Gurobi" if formulation == "C" else "VersionMN+Gurobi"
        raise ValueError(f"Unsupported solid solver backend: {backend!r}.")

    if family == "shell":
        if shell_yield not in {"ilyushin", "layer"}:
            raise ValueError(f"Shell shell_yield must be Ilyushin or layer, got {shell_yield!r}.")
        if backend == "cvxpy":
            return "VersionC_Ilyushin" if shell_yield == "ilyushin" else "VersionC_shell"
        if backend == "gurobi":
            return "VersionC_Ilyushin+Gurobi" if shell_yield == "ilyushin" else "VersionC_shell+Gurobi"
        raise ValueError(f"Unsupported shell solver backend: {backend!r}.")

    raise ValueError(f"Cannot infer shakedown family from MAT files: {family!r}.")


def parse_shakedown_config(config_path: str | Path | None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if config_path is None:
        raise ValueError("Shakedown config must define mat_files.")
    config_path = Path(config_path).resolve()
    config["config_path"] = str(config_path)
    config["workflow_dir"] = str(config_path.parent)

    for key, value in _collect_cfg_items(config_path):
        if not key:
            continue
        if key == "mat_files":
            config[key] = parse_mat_files(value)
        elif key == "formulation":
            config[key] = value
        elif key == "shell_yield":
            config[key] = value
        elif key == "ilyushin_c":
            config[key] = float(value)
        elif key == "solver":
            config["solver_backend"] = value
        elif key == "gurobi_backend":
            config[key] = value
        elif key == "windows_gurobi_root":
            config[key] = value
        elif key == "windows_python_exe":
            config[key] = value
        elif key == "windows_matlab_exe":
            config[key] = value
        elif key == "windows_gurobi_user":
            config[key] = value
        elif key == "windows_gurobi_password":
            config[key] = value
        elif key == "gurobi_params":
            config[key] = parse_literal_dict(value)
        elif key == "wsl_python":
            config[key] = value
        elif key == "R_ratios":
            config[key] = parse_ratio_array(value)
        elif key == "use_angles":
            config[key] = parse_bool(value)
        elif key == "theta_deg":
            config[key] = parse_angle_array(value)
        elif key == "phi_deg":
            config[key] = parse_angle_array(value)
        elif key == "load_factor_set":
            # Store raw string; parse after mat_files is resolved (n_cases known)
            config["_lfs_raw"] = value
        elif key == "yield_values":
            config[key] = parse_yield_values(value)
        elif key == "num_vert":
            config[key] = int(value)
        else:
            config[key] = value

    if not config["mat_files"]:
        raise ValueError("Shakedown config must define mat_files.")
    if not (-1.0 <= float(config["ilyushin_c"]) <= 1.0):
        raise ValueError(f"ilyushin_c must be within [-1, 1], got {config['ilyushin_c']}.")
    config["load_cases"] = [Path(path).stem for path in config["mat_files"]]
    config["family"] = _infer_family_from_mat(config["mat_files"][0])
    config["solver"] = _resolve_solver_key(config)

    n_cases = len(config["mat_files"])

    # Pad R_ratios with zeros to match n_cases
    r = config["R_ratios"]
    if len(r) < n_cases:
        config["R_ratios"] = r + [0.0] * (n_cases - len(r))
    elif len(r) > n_cases:
        raise ValueError(
            f"R_ratios has {len(r)} values but {n_cases} MAT files are specified."
        )

    # Parse load_factor_set now that n_cases is known
    if "_lfs_raw" in config:
        config["load_factor_set"] = parse_load_factor_set(config.pop("_lfs_raw"), n_cases)
    else:
        config.pop("_lfs_raw", None)

    if int(config["num_vert"]) not in {1, 2, 4, 8}:
        raise ValueError(f"num_vert must be one of {{1,2,4,8}}, got {config['num_vert']}.")
    if int(config["num_vert"]) not in {1, 2} and int(config["num_vert"]) != 2 ** n_cases:
        raise ValueError(
            f"For {n_cases} load cases, num_vert must be 1, 2, or {2 ** n_cases}; got {config['num_vert']}."
        )
    if not config["use_angles"]:
        if config["load_factor_set"] is None:
            raise ValueError("When use_angles = no, load_factor_set must be provided.")
    else:
        config["load_factor_set"] = None

    # Auto-fill machine-level keys from env.cfg if not set in the shakedown cfg
    _MACHINE_KEYS = [
        "windows_python_exe",
        "windows_matlab_exe",
        "windows_gurobi_root",
        "windows_gurobi_user",
        "windows_gurobi_password",
        "wsl_python",
    ]
    for _k in _MACHINE_KEYS:
        if not str(config.get(_k, "")).strip():
            _v = _MACHINE_CONFIG.get(_k, "")
            if _v:
                config[_k] = _v

    return config

