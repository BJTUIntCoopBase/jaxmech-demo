"""Discover demo models and the supported workflow outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "Examples"
STORED_MODELS_DIR = PROJECT_ROOT / "StoredModels"


def _file_info(path: Path) -> dict[str, Any]:
    return {"name": path.name, "mtime": int(path.stat().st_mtime)}


def _workflow_mats(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*.mat")
        if not any(part.lower() == "inputs" for part in path.relative_to(root).parts)
    )


def _model_status(model_dir: Path) -> dict[str, Any]:
    """Inspect one model directory for the demo-supported workflows only."""
    abaqus_dir = model_dir / "abaqus"
    inc_dir = model_dir / "inc_analysis"
    val_dir = model_dir / "validation"
    sd_dir = model_dir / "shakedown"

    inp_files = sorted(abaqus_dir.glob("*.inp")) if abaqus_dir.is_dir() else []
    inc_mats = _workflow_mats(inc_dir)
    val_mats = sorted(val_dir.rglob("*.mat")) if val_dir.is_dir() else []
    sd_mats = sorted(sd_dir.rglob("*.mat")) if sd_dir.is_dir() else []
    inc_names = {path.name for path in inc_mats}
    validated_names = inc_names & {path.name for path in val_mats}

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "abaqus": {
            "inp_count": len(inp_files),
            "inp_files": [path.name for path in inp_files],
        },
        "inc_analysis": {
            "has_mat": bool(inc_mats),
            "mat_files": [str(path.relative_to(inc_dir)) for path in inc_mats],
            "mat_files_info": [_file_info(path) for path in inc_mats],
            "has_cfg": bool(list(inc_dir.glob("*.cfg"))) if inc_dir.is_dir() else False,
        },
        "validation": {
            "validated": bool(validated_names) and len(validated_names) == len(inc_names) if inc_names else False,
            "matched_mat_count": len(validated_names),
            "mat_files": [str(path.relative_to(val_dir)) for path in val_mats] if val_dir.is_dir() else [],
            "mat_files_info": [_file_info(path) for path in val_mats],
            "has_cfg": bool(list(val_dir.glob("*.cfg"))) if val_dir.is_dir() else False,
        },
        "shakedown": {
            "has_results": bool(sd_mats),
            "mat_files": [path.name for path in sd_mats],
            "mat_files_info": [_file_info(path) for path in sd_mats],
            "has_cfg": bool(list(sd_dir.glob("*.cfg"))) if sd_dir.is_dir() else False,
        },
    }


def scan_models() -> list[dict[str, Any]]:
    """Scan Examples/ and StoredModels/ for demo model directories."""
    models: list[dict[str, Any]] = []
    for root_dir, source in ((EXAMPLES_DIR, "examples"), (STORED_MODELS_DIR, "stored")):
        if not root_dir.is_dir():
            continue
        for child in sorted(root_dir.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                info = _model_status(child)
                info["source"] = source
                models.append(info)
    return models


def get_model_detail(source: str, name: str) -> dict[str, Any] | None:
    """Return detailed status for one model."""
    root = EXAMPLES_DIR if source == "examples" else STORED_MODELS_DIR
    model_dir = root / name
    if not model_dir.is_dir():
        return None
    info = _model_status(model_dir)
    info["source"] = source
    return info
