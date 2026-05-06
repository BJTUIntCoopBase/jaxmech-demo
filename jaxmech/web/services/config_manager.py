"""Read and write demo configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jaxmech.env.env import DEFAULT_MACHINE_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_CFG = PROJECT_ROOT / "Config" / "env.cfg"
ENV_TEMPLATE = PROJECT_ROOT / "Config" / "env.template.cfg"

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
}


def read_env_cfg() -> dict[str, Any]:
    cfg_path = ENV_CFG if ENV_CFG.is_file() else ENV_TEMPLATE
    values = dict(_parse_cfg(cfg_path).get("values") or {})
    return {"values": _apply_env_defaults(values)}


def write_env_cfg(data: dict[str, str]) -> None:
    cfg_path = ENV_CFG
    if not cfg_path.is_file():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(ENV_TEMPLATE.read_text(encoding="utf-8") if ENV_TEMPLATE.is_file() else "", encoding="utf-8")

    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    existing: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in data:
                out.append(f"{key} = {data[key]}")
                existing.add(key)
            else:
                out.append(line)
        else:
            out.append(line)
    for key, value in data.items():
        if key not in existing:
            out.append(f"{key} = {value}")
    cfg_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def read_module_cfg(module: str, model_path: str) -> dict[str, Any]:
    if module not in MODULE_TEMPLATES:
        return {"source": "none", "path": "", "content": "", "values": {}}
    module_dir = Path(model_path) / module
    cfg_files = sorted(module_dir.glob("*.cfg")) if module_dir.is_dir() else []
    if cfg_files:
        cfg_path = cfg_files[0]
        return {"source": "custom", "path": str(cfg_path), "content": cfg_path.read_text(encoding="utf-8"), **_parse_cfg(cfg_path)}
    template_path = MODULE_TEMPLATES[module][0]
    if template_path.is_file():
        return {"source": "template", "path": str(template_path), "content": template_path.read_text(encoding="utf-8"), **_parse_cfg(template_path)}
    return {"source": "none", "path": "", "content": "", "values": {}}


def write_module_cfg(module: str, model_path: str, content: str) -> str:
    if module not in MODULE_TEMPLATES:
        raise ValueError(f"Unsupported demo module: {module}")
    module_dir = Path(model_path)
    module_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = module_dir / MODULE_TEMPLATES[module][1]
    cfg_path.write_text(content, encoding="utf-8")
    return str(cfg_path)


def _parse_cfg(cfg_path: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    if not cfg_path.is_file():
        return {"values": values}
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
        tail = text[7:].replace("/", "\\")
        return f"{text[5].upper()}:\\{tail}"
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
    return merged
