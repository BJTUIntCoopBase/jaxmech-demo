"""Config manager — read/write cfg files with template fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jaxmech.env.env import DEFAULT_MACHINE_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_CFG = PROJECT_ROOT / "Config" / "env.cfg"
ENV_TEMPLATE = PROJECT_ROOT / "Config" / "env.template.cfg"

# Map from module key to (template path, default cfg filename)
MODULE_TEMPLATES: dict[str, tuple[Path, str]] = {
    "inc_analysis": (
        PROJECT_ROOT / "jaxmech" / "modules" / "inc_analysis" / "templates" / "inc_analysis.template.cfg",
        "inc_analysis.template.cfg",
    ),
    "validation": (
        PROJECT_ROOT / "jaxmech" / "modules" / "validation" / "elastic" / "templates" / "validation.template.cfg",
        "validation.template.cfg",
    ),
    "shakedown": (
        PROJECT_ROOT / "jaxmech" / "modules" / "shakedown" / "templates" / "shakedown_analysis.template.cfg",
        "shakedown_analysis.template.cfg",
    ),
    "dca": (
        PROJECT_ROOT / "jaxmech" / "modules" / "direct_methods" / "dca" / "templates" / "dca.template.cfg",
        "dca.template.cfg",
    ),
}


def read_env_cfg() -> dict[str, Any]:
    """Read Config/env.cfg and return key=value pairs."""
    cfg_path = ENV_CFG if ENV_CFG.is_file() else ENV_TEMPLATE
    parsed = _parse_cfg(cfg_path)
    values = dict(parsed.get("values") or {})
    values = _apply_env_defaults(values)
    return {"values": values}


def write_env_cfg(data: dict[str, str]) -> None:
    """Write key=value pairs back to Config/env.cfg, preserving comments."""
    cfg_path = ENV_CFG
    if not cfg_path.is_file():
        # Copy template first
        if ENV_TEMPLATE.is_file():
            cfg_path.write_text(ENV_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            cfg_path.write_text("", encoding="utf-8")

    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    existing_keys: set[str] = set()
    new_lines: list[str] = []

    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in data:
                new_lines.append(f"{key} = {data[key]}")
                existing_keys.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append any keys not already in the file
    for key, value in data.items():
        if key not in existing_keys:
            new_lines.append(f"{key} = {value}")

    cfg_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def read_module_cfg(module: str, model_path: str) -> dict[str, Any]:
    """Read a module-level cfg from a model directory, with template fallback."""
    model_dir = Path(model_path)
    module_dir = model_dir / module

    # Look for existing cfg
    cfg_files = list(module_dir.glob("*.cfg")) if module_dir.is_dir() else []
    if cfg_files:
        return {
            "source": "custom",
            "path": str(cfg_files[0]),
            "content": cfg_files[0].read_text(encoding="utf-8"),
            **_parse_cfg(cfg_files[0]),
        }

    # Fallback to template
    if module in MODULE_TEMPLATES:
        template_path = MODULE_TEMPLATES[module][0]
        if template_path.is_file():
            return {
                "source": "template",
                "path": str(template_path),
                "content": template_path.read_text(encoding="utf-8"),
                **_parse_cfg(template_path),
            }

    return {"source": "none", "path": "", "content": "", "values": {}}


def write_module_cfg(module: str, model_path: str, content: str) -> str:
    """Write a module cfg into the model's module directory.

    For shakedown: if model_path already contains a 'shakedown' path segment
    (i.e. the caller is passing a timestamped sub-directory), write the cfg
    directly into that directory rather than creating an extra shakedown/ layer.
    """
    model_dir = Path(model_path)

    if module == "shakedown" and "shakedown" in [p.lower() for p in model_dir.parts]:
        # model_path is already the shakedown sub-directory (e.g. …/shakedown/20260403_102556)
        module_dir = model_dir
    else:
        module_dir = model_dir / module

    module_dir.mkdir(parents=True, exist_ok=True)

    if module in MODULE_TEMPLATES:
        filename = MODULE_TEMPLATES[module][1]
    else:
        filename = f"{module}.cfg"

    cfg_path = module_dir / filename
    cfg_path.write_text(content, encoding="utf-8")
    return str(cfg_path)


def _parse_cfg(cfg_path: Path) -> dict[str, Any]:
    """Parse a simple key=value cfg file, skipping comments."""
    values: dict[str, str] = {}
    for line in cfg_path.read_text(encoding="utf-8").splitlines():
        data = line.split("#", 1)[0].strip()
        if "=" in data:
            key, value = data.split("=", 1)
            values[key.strip()] = value.strip()
    return {"values": values}


def _win_to_wsl_text(path_text: str) -> str:
    text = str(path_text).strip().replace("\\", "/")
    if len(text) >= 3 and text[1] == ":" and text[2] == "/":
        return f"/mnt/{text[0].lower()}/{text[3:]}"
    return text


def _wsl_to_win_text(path_text: str) -> str:
    text = str(path_text).strip().replace("\\", "/")
    if text.startswith("/mnt/") and len(text) > 7 and text[6] == "/":
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        return f"{drive}:\\{tail}" if tail else f"{drive}:\\"
    return text.replace("/", "\\")


def _apply_env_defaults(values: dict[str, str]) -> dict[str, str]:
    merged = dict(DEFAULT_MACHINE_CONFIG)
    explicit_win_root = str(values.get("win_stored_root", "")).strip()
    explicit_wsl_root = str(values.get("wsl_stored_root", "")).strip()
    for key, value in values.items():
        if str(value).strip():
            merged[key] = str(value).strip()

    if explicit_win_root and not explicit_wsl_root:
        merged["wsl_stored_root"] = _win_to_wsl_text(explicit_win_root)
    elif explicit_wsl_root and not explicit_win_root:
        merged["win_stored_root"] = _wsl_to_win_text(explicit_wsl_root)
    elif explicit_win_root and explicit_wsl_root:
        if explicit_wsl_root == DEFAULT_MACHINE_CONFIG.get("wsl_stored_root", "") and explicit_win_root != DEFAULT_MACHINE_CONFIG.get("win_stored_root", ""):
            merged["wsl_stored_root"] = _win_to_wsl_text(explicit_win_root)
        elif explicit_win_root == DEFAULT_MACHINE_CONFIG.get("win_stored_root", "") and explicit_wsl_root != DEFAULT_MACHINE_CONFIG.get("wsl_stored_root", ""):
            merged["win_stored_root"] = _wsl_to_win_text(explicit_wsl_root)
    return merged
