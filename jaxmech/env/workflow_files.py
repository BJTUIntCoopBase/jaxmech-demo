"""Workflow-directory and template discovery helpers for the demo subset."""

from __future__ import annotations

from pathlib import Path

from jaxmech.env.env import PROJECT_ROOT


WORKFLOW_KEYWORDS = {
    "inc_analysis": ("inc_analysis",),
    "validation": ("validation",),
    "shakedown": ("shakedown",),
}

DEFAULT_TEMPLATE_PATHS = {
    "inc_analysis": PROJECT_ROOT / "jaxmech" / "modules" / "inc_analysis" / "templates" / "inc_analysis.template.cfg",
    "validation": PROJECT_ROOT / "jaxmech" / "modules" / "validation" / "elastic" / "templates" / "validation.template.cfg",
    "shakedown": PROJECT_ROOT / "jaxmech" / "modules" / "shakedown" / "templates" / "shakedown_analysis.template.cfg",
}


def _matches_workflow(name: str, workflow: str) -> bool:
    return any(token in str(name).lower() for token in WORKFLOW_KEYWORDS[workflow])


def _choose_cfg_file(workflow_dir: Path, workflow: str) -> Path | None:
    cfg_files = sorted(workflow_dir.glob("*.cfg"))
    if not cfg_files:
        return None
    preferred = [path for path in cfg_files if "template" in path.stem.lower() or _matches_workflow(path.stem, workflow)]
    if preferred:
        return sorted(preferred)[0]
    if len(cfg_files) == 1:
        return cfg_files[0]
    raise ValueError(f"Multiple cfg files found in {workflow_dir} for workflow {workflow!r}.")


def resolve_workflow_target(target: str | Path, workflow: str) -> dict[str, Path | None]:
    workflow = str(workflow).strip().lower()
    if workflow not in WORKFLOW_KEYWORDS:
        raise ValueError(f"Unsupported demo workflow: {workflow!r}")

    path = Path(target).resolve()
    if path.is_file():
        return {
            "workflow_dir": path.parent,
            "config_path": path,
            "model_dir": path.parent.parent,
            "default_template_path": DEFAULT_TEMPLATE_PATHS[workflow],
        }
    if not path.is_dir():
        raise FileNotFoundError(f"Workflow target not found: {path}")

    workflow_dir = path
    if not _matches_workflow(path.name, workflow):
        candidates = sorted(child for child in path.iterdir() if child.is_dir() and _matches_workflow(child.name, workflow))
        if not candidates:
            raise ValueError(f"No subdirectory matching workflow {workflow!r} found under {path}.")
        if len(candidates) > 1:
            raise ValueError(f"Multiple subdirectories match workflow {workflow!r} under {path}.")
        workflow_dir = candidates[0]

    return {
        "workflow_dir": workflow_dir,
        "config_path": _choose_cfg_file(workflow_dir, workflow),
        "model_dir": workflow_dir.parent,
        "default_template_path": DEFAULT_TEMPLATE_PATHS[workflow],
    }


def find_unique_workflow_dir(model_dir: str | Path, workflow: str) -> Path | None:
    workflow = str(workflow).strip().lower()
    if workflow not in WORKFLOW_KEYWORDS:
        raise ValueError(f"Unsupported demo workflow: {workflow!r}")
    model_path = Path(model_dir).resolve()
    candidates = sorted(child for child in model_path.iterdir() if child.is_dir() and _matches_workflow(child.name, workflow))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple sibling workflow directories match {workflow!r} under {model_path}.")
    return candidates[0]
