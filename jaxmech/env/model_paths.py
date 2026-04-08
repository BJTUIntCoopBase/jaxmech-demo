"""Resolve model-specific INP / MAT / ODB paths from configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

from jaxmech.env.env import PROJECT_ROOT


EXAMPLES_ROOT = PROJECT_ROOT / "Examples"
LOCAL_MODELS_ROOT = PROJECT_ROOT / "StoredModels"


def normalize_jax_system(value: str) -> str:
    norm = str(value).strip().upper()
    if norm not in {"WSL", "WIN"}:
        raise ValueError(f"Unsupported jax_system: {value}")
    return norm


def win_to_wsl_path(path_str: str) -> Path:
    s = str(path_str).strip()
    if s.startswith("/"):
        return Path(s)
    if len(s) >= 3 and s[1] == ":" and s[2] in ("\\", "/"):
        drive = s[0].lower()
        tail = s[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail
    return Path(s)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        deduped.append(path)
        seen.add(key)
    return deduped


def _candidate_model_dirs(config: dict) -> list[Path]:
    jax_system = normalize_jax_system(config["jax_system"])
    primary = (
        Path(config["wsl_stored_root"])
        if jax_system == "WSL"
        else win_to_wsl_path(config["win_stored_root"])
    )
    secondary = (
        win_to_wsl_path(config["win_stored_root"])
        if jax_system == "WSL"
        else Path(config["wsl_stored_root"])
    )
    model_name = str(config["model_name"])
    dirs = [
        primary / model_name,
        secondary / model_name,
        LOCAL_MODELS_ROOT / model_name,
        EXAMPLES_ROOT / model_name,
    ]
    return _dedupe_paths(dirs)


def _search_case_file(case_name: str, suffix: str, search_dirs: list[Path]) -> Path | None:
    rels = [
        Path(f"{case_name}{suffix}"),
        Path("jax") / f"{case_name}{suffix}",
        Path("JAX") / f"{case_name}{suffix}",
        Path("abaqus") / f"{case_name}{suffix}",
        Path("ABAQUS") / f"{case_name}{suffix}",
    ]
    for base in search_dirs:
        for rel in rels:
            candidate = base / rel
            if candidate.is_file():
                return candidate
    return None


def _classify_model_dir(model_dir: Path) -> None:
    """Normalize one model directory to the standard abaqus/jax layout."""
    if not model_dir.is_dir():
        return

    abaqus_dir = model_dir / "abaqus"
    jax_dir = model_dir / "jax"

    for legacy_name, canonical_dir in (("ABAQUS", abaqus_dir), ("JAX", jax_dir)):
        legacy_dir = model_dir / legacy_name
        if legacy_dir.is_dir():
            canonical_dir.mkdir(parents=True, exist_ok=True)
            for child in legacy_dir.iterdir():
                target = canonical_dir / child.name
                if target.exists():
                    continue
                shutil.move(str(child), str(target))
            try:
                legacy_dir.rmdir()
            except OSError:
                pass

    for path in model_dir.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".inp", ".odb"}:
            abaqus_dir.mkdir(parents=True, exist_ok=True)
            target = abaqus_dir / path.name
        elif suffix == ".mat":
            jax_dir.mkdir(parents=True, exist_ok=True)
            target = jax_dir / path.name
        else:
            continue

        if target == path or target.exists():
            continue
        shutil.move(str(path), str(target))


def resolve_model_case_paths(config: dict, case_name: str) -> dict[str, Path]:
    model_dirs = _candidate_model_dirs(config)
    jax_system = normalize_jax_system(config["jax_system"])
    out_root = (
        Path(config["wsl_stored_root"])
        if jax_system == "WSL"
        else win_to_wsl_path(config["win_stored_root"])
    ) / str(config["model_name"])

    out_root.parent.mkdir(parents=True, exist_ok=True)
    _classify_model_dir(out_root)

    out_abaqus_dir = out_root / "abaqus"
    out_jax_dir = out_root / "jax"
    out_abaqus_dir.mkdir(parents=True, exist_ok=True)
    out_jax_dir.mkdir(parents=True, exist_ok=True)

    inp_path = _search_case_file(case_name, ".inp", model_dirs)
    odb_path = _search_case_file(case_name, ".odb", model_dirs)
    mat_path = _search_case_file(case_name, ".mat", model_dirs)

    return {
        "model_dir": out_root,
        "abaqus_dir": out_abaqus_dir,
        "jax_dir": out_jax_dir,
        "inp_path": inp_path if inp_path is not None else (out_abaqus_dir / f"{case_name}.inp"),
        "odb_path": odb_path if odb_path is not None else (out_abaqus_dir / f"{case_name}.odb"),
        "mat_path": mat_path if mat_path is not None else (out_jax_dir / f"{case_name}.mat"),
    }
