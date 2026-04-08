"""Model scanner — discover example/stored models and their status."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = PROJECT_ROOT / "Examples"
STORED_MODELS_DIR = PROJECT_ROOT / "StoredModels"


def _model_status(model_dir: Path) -> dict[str, Any]:
    """Inspect one model directory and report per-module status."""
    abaqus_dir = model_dir / "abaqus"
    inc_dir = model_dir / "inc_analysis"
    dca_dir = model_dir / "dca"
    val_dir = model_dir / "validation"
    sd_dir = model_dir / "shakedown"

    inp_files = sorted(abaqus_dir.glob("*.inp")) if abaqus_dir.is_dir() else []
    odb_files = sorted(abaqus_dir.glob("*.odb")) if abaqus_dir.is_dir() else []
    mat_files = sorted(inc_dir.glob("*.mat")) if inc_dir.is_dir() else []
    dca_mats = sorted(dca_dir.glob("*.mat")) if dca_dir.is_dir() else []
    val_mats = sorted(val_dir.glob("*.mat")) if val_dir.is_dir() else []
    sd_mats = sorted(sd_dir.glob("*.mat")) if sd_dir.is_dir() else []

    inc_names = {f.name for f in mat_files}
    validated_names = inc_names & {f.name for f in val_mats}

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "abaqus": {
            "inp_count": len(inp_files),
            "odb_count": len(odb_files),
            "inp_files": [f.name for f in inp_files],
            "odb_files": [f.name for f in odb_files],
        },
        "inc_analysis": {
            "has_mat": bool(mat_files),
            "mat_files": [f.name for f in mat_files],
            "mat_files_info": [
                {"name": f.name, "mtime": int(f.stat().st_mtime)}
                for f in mat_files
            ],
            "has_cfg": bool(list(inc_dir.glob("*.cfg"))) if inc_dir.is_dir() else False,
        },
        "dca": {
            "has_mat": bool(dca_mats),
            "mat_files": [f.name for f in dca_mats],
            "mat_files_info": [
                {"name": f.name, "mtime": int(f.stat().st_mtime)}
                for f in dca_mats
            ],
            "has_cfg": bool(list(dca_dir.glob("*.cfg"))) if dca_dir.is_dir() else False,
        },
        "validation": {
            "validated": bool(validated_names) and len(validated_names) == len(inc_names) if inc_names else False,
            "matched_mat_count": len(validated_names),
            "mat_files": [f.name for f in val_mats],
            "has_cfg": bool(list(val_dir.glob("*.cfg"))) if val_dir.is_dir() else False,
        },
        "shakedown": {
            "has_results": bool(sd_mats),
            "mat_files": [f.name for f in sd_mats],
            "has_cfg": bool(list(sd_dir.glob("*.cfg"))) if sd_dir.is_dir() else False,
        },
    }


def scan_models() -> list[dict[str, Any]]:
    """Scan both Examples/ and StoredModels/ for model directories."""
    models: list[dict[str, Any]] = []
    for root_dir, source in [(EXAMPLES_DIR, "examples"), (STORED_MODELS_DIR, "stored")]:
        if not root_dir.is_dir():
            continue
        for child in sorted(root_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            info = _model_status(child)
            info["source"] = source
            models.append(info)
    return models


def get_model_detail(source: str, name: str) -> dict[str, Any] | None:
    """Get detailed status for one specific model."""
    root = EXAMPLES_DIR if source == "examples" else STORED_MODELS_DIR
    model_dir = root / name
    if not model_dir.is_dir():
        return None
    info = _model_status(model_dir)
    info["source"] = source
    return info
