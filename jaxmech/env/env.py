"""Repository-level machine configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "Config"
ENV_TEMPLATE = CONFIG_DIR / "env.template.cfg"
ENV_CFG = CONFIG_DIR / "env.cfg"


def _win_path_to_wsl_text(path: Path | str) -> str:
    text = str(path).strip()
    if text.startswith("/"):
        return text.replace("\\", "/")
    if len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/"):
        drive = text[0].lower()
        tail = text[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{tail}"
    return text.replace("\\", "/")


def _wsl_path_to_win_text(path: Path | str) -> str:
    text = str(path).strip().replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    return text.replace("/", "\\")


_DEFAULT_STORED_ROOT_WIN = str(PROJECT_ROOT / "StoredModels")
_DEFAULT_STORED_ROOT_WSL = _win_path_to_wsl_text(_DEFAULT_STORED_ROOT_WIN)


DEFAULT_MACHINE_CONFIG = {
    "wsl_python": "",
    "windows_python_exe": "",
    "win_stored_root": _DEFAULT_STORED_ROOT_WIN,
    "wsl_stored_root": _DEFAULT_STORED_ROOT_WSL,
}


def _iter_cfg_keys(lines: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for line in lines:
        data = line.split("#", 1)[0].strip()
        if not data or "=" not in data:
            continue
        key = data.split("=", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def _append_missing_template_keys() -> None:
    """Append missing template keys into an existing ``env.cfg``."""
    if not ENV_TEMPLATE.exists() or not ENV_CFG.exists():
        return

    template_lines = ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    existing_lines = ENV_CFG.read_text(encoding="utf-8").splitlines()
    existing_keys = set(_iter_cfg_keys(existing_lines))

    pending_blocks: list[str] = []
    pending_comments: list[str] = []
    for line in template_lines:
        stripped = line.strip()
        if not stripped:
            pending_comments = []
            continue
        if stripped.startswith("#"):
            pending_comments.append(line)
            continue
        if "=" not in line:
            pending_comments = []
            continue

        key = line.split("=", 1)[0].strip()
        if key and key not in existing_keys:
            if pending_comments:
                pending_blocks.extend(pending_comments)
            pending_blocks.append(line)
            pending_blocks.append("")
            existing_keys.add(key)
        pending_comments = []

    if pending_blocks:
        current = ENV_CFG.read_text(encoding="utf-8")
        suffix = "\n".join(pending_blocks).rstrip() + "\n"
        if current and not current.endswith("\n"):
            current += "\n"
        ENV_CFG.write_text(current + "\n" + suffix, encoding="utf-8")


def ensure_env_cfg_exists() -> Path | None:
    """Create ``Config/env.cfg`` from template if possible and missing."""
    if not ENV_CFG.exists():
        if not ENV_TEMPLATE.exists():
            return None
        ENV_CFG.write_text(ENV_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    _append_missing_template_keys()
    return ENV_CFG


def load_machine_config() -> dict:
    """Load machine-level config from ``Config/env.cfg`` with safe defaults."""
    config = dict(DEFAULT_MACHINE_CONFIG)
    env_path = ensure_env_cfg_exists()
    if env_path is None or not env_path.exists():
        return config

    loaded_values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        data = line.split("#", 1)[0].strip()
        if not data or "=" not in data:
            continue
        key, value = [item.strip() for item in data.split("=", 1)]
        if not key:
            continue
        loaded_values[key] = value

    config.update(loaded_values)

    explicit_win_root = str(loaded_values.get("win_stored_root", "")).strip()
    explicit_wsl_root = str(loaded_values.get("wsl_stored_root", "")).strip()
    if explicit_win_root and not explicit_wsl_root:
        config["wsl_stored_root"] = _win_path_to_wsl_text(explicit_win_root)
    elif explicit_wsl_root and not explicit_win_root:
        config["win_stored_root"] = _wsl_path_to_win_text(explicit_wsl_root)
    elif explicit_win_root and explicit_wsl_root:
        if explicit_wsl_root == _DEFAULT_STORED_ROOT_WSL and explicit_win_root != _DEFAULT_STORED_ROOT_WIN:
            config["wsl_stored_root"] = _win_path_to_wsl_text(explicit_win_root)
        elif explicit_win_root == _DEFAULT_STORED_ROOT_WIN and explicit_wsl_root != _DEFAULT_STORED_ROOT_WSL:
            config["win_stored_root"] = _wsl_path_to_win_text(explicit_wsl_root)
    return config
