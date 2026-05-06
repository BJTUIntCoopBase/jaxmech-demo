"""Models API for the demo-supported workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jaxmech.io.abaqus.inp_metadata import parse_inp_metadata_batch
from jaxmech.io.abaqus.inp_metadata_store import (
    decode_analysis_input_bundle,
    save_analysis_input_snapshot,
)
from jaxmech.web.services.model_scanner import get_model_detail, scan_models


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models() -> list[dict[str, Any]]:
    return scan_models()


@router.get("/{source}/{name}")
async def model_detail(source: str, name: str) -> dict[str, Any]:
    info = get_model_detail(source, name)
    if info is None:
        raise HTTPException(404, f"Model not found: {source}/{name}")
    return info


class ScanMatsRequest(BaseModel):
    model_path: str


class InpMetaRequest(BaseModel):
    paths: list[str]


class PrepareIncAnalysisRequest(BaseModel):
    model_path: str
    inp_paths: list[str]
    analysis_input: dict[str, Any]
    source: str = "web_inc_analysis"


class MatMetaRequest(BaseModel):
    paths: list[str]


class FindOdbRequest(BaseModel):
    mat_path: str
    model_path: str = ""


def _unwrap_scalar(value: Any) -> Any:
    cur = value
    while isinstance(cur, np.ndarray) and cur.size == 1:
        cur = cur.reshape(-1)[0]
    return cur


def _struct_to_dict(value: Any) -> dict[str, Any]:
    cur = _unwrap_scalar(value)
    if isinstance(cur, dict):
        return {str(key): _unwrap_scalar(val) for key, val in cur.items()}
    if hasattr(cur, "_fieldnames"):
        return {str(name): _unwrap_scalar(getattr(cur, name)) for name in getattr(cur, "_fieldnames", [])}
    if isinstance(cur, np.ndarray) and getattr(cur.dtype, "names", None):
        rec = cur.reshape(-1)[0]
        return {str(name): _unwrap_scalar(rec[name]) for name in cur.dtype.names or []}
    return {}


def _mat_metadata_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return _struct_to_dict(payload.get("metadata"))


def _clean_mat_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: _unwrap_scalar(value) for key, value in raw.items() if not key.startswith("__")}


def _string_field(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    if value is None:
        return default
    return str(_unwrap_scalar(value)).strip() or default


def _inp_path_from_payload(payload: dict[str, Any], metadata: dict[str, Any]) -> str:
    inp_data = _struct_to_dict(payload.get("InpData"))
    if inp_data.get("inp_path"):
        return str(inp_data["inp_path"]).strip()
    for key in ("source_inp", "source_file"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _check_validated(payload: dict[str, Any]) -> tuple[bool, bool]:
    present = {key for key in payload if not key.startswith("__")}
    if "validated_with_ODB" in present:
        try:
            val = np.asarray(payload["validated_with_ODB"], dtype=np.int32).reshape(-1)[0]
            return int(val) == 1, True
        except Exception:
            return bool(payload["validated_with_ODB"]), True
    for key in ("validated", "validation_passed", "odb_validation", "is_validated"):
        if key in present:
            try:
                val = np.asarray(payload[key]).reshape(-1)[0]
                return bool(val), True
            except Exception:
                return bool(payload[key]), True
    return False, False


@router.post("/scan-mats")
async def scan_mats(req: ScanMatsRequest) -> dict[str, Any]:
    model_dir = Path(req.model_path)
    if not model_dir.is_dir():
        raise HTTPException(404, f"Model directory not found: {req.model_path}")
    results = []
    for root_name in ("inc_analysis", "validation", "shakedown"):
        root = model_dir / root_name
        if not root.is_dir():
            continue
        for mat_path in sorted(root.rglob("*.mat")):
            rel = mat_path.relative_to(model_dir)
            results.append({
                "abs_path": str(mat_path),
                "rel_path": str(rel),
                "name": mat_path.name,
                "subdir": str(rel.parent),
            })
    return {"model_path": str(model_dir), "mat_files": results}


@router.post("/inp-meta")
async def inp_meta(req: InpMetaRequest) -> dict[str, Any]:
    results = parse_inp_metadata_batch(req.paths)
    return {"count": len(results), "details": results}


@router.post("/prepare-inc-analysis")
async def prepare_inc_analysis(req: PrepareIncAnalysisRequest) -> dict[str, Any]:
    model_dir = Path(req.model_path)
    if not model_dir.is_dir():
        raise HTTPException(404, f"Model directory not found: {req.model_path}")
    if not req.inp_paths:
        raise HTTPException(400, "No INP files were provided.")

    results: list[dict[str, Any]] = []
    for raw_path in req.inp_paths:
        inp_path = Path(raw_path)
        if not inp_path.is_file():
            raise HTTPException(404, f"INP file not found: {raw_path}")
        try:
            inp_path.resolve().relative_to(model_dir.resolve())
        except Exception:
            raise HTTPException(400, f"INP file is outside model directory: {raw_path}")
        inp_meta = parse_inp_metadata_batch([str(inp_path)])[0]
        if inp_meta.get("error"):
            raise HTTPException(422, f"Failed to parse {inp_path.name}: {inp_meta['error']}")
        if inp_meta.get("family") != "solid":
            raise HTTPException(422, f"Only solid INP files are supported in the demo: {inp_path.name}")

        result_mat_path = model_dir / "inc_analysis" / f"{inp_path.stem}.mat"
        status = save_analysis_input_snapshot(result_mat_path, inp_meta, req.analysis_input, source=req.source)
        results.append({
            "inp_path": str(inp_path),
            "inp_name": inp_path.name,
            "result_mat_path": status["result_mat_path"],
            "sidecar_mat_path": status["sidecar_mat_path"],
            "changed_fields": status["changed_fields"],
            "history_count": status["history_count"],
            "updated_result_mat": status["updated_result_mat"],
            "analysis_input": status["analysis_input"],
        })

    return {
        "count": len(results),
        "workflow_dir": str(model_dir / "inc_analysis"),
        "sidecar_dir": str(model_dir / "inc_analysis" / "inputs"),
        "details": results,
    }


@router.post("/mat-meta")
async def mat_meta(req: MatMetaRequest) -> dict[str, Any]:
    try:
        import scipy.io as sio
    except ImportError:
        raise HTTPException(500, "scipy is not available in Windows Python.")

    details: list[dict[str, Any]] = []
    for raw_path in req.paths:
        path = Path(raw_path)
        if not path.is_file():
            raise HTTPException(404, f"MAT file not found: {raw_path}")
        try:
            payload = _clean_mat_payload(sio.loadmat(str(path), squeeze_me=False, struct_as_record=False))
        except Exception as exc:
            raise HTTPException(422, f"Failed to read {path.name}: {exc}")

        metadata = _mat_metadata_dict(payload)
        bundle = decode_analysis_input_bundle(payload)
        present = {key for key in payload if not key.startswith("__")}
        family = str(metadata.get("family") or _string_field(payload, "family", "solid")).lower()
        has_c = "C_sparse" in present
        is_shakedown = "ResultsSet" in present or "ElasticInputSet" in present or "residual_stress" in present
        analysis_type = "shakedown" if is_shakedown else str(metadata.get("analysis_type") or "linear_static")
        frame_count = int(np.asarray(payload.get("frame_u", [])).shape[0]) if "frame_u" in present else 1
        has_frame_history = frame_count > 1
        is_validated, has_validation_flag = _check_validated(payload)
        details.append({
            "path": raw_path,
            "name": path.name,
            "family": family,
            "is_validated": is_validated,
            "has_validation_flag": has_validation_flag,
            "has_required_fields": bool(has_c or is_shakedown),
            "missing_fields": [] if has_c or is_shakedown else ["C_sparse"],
            "all_fields": sorted(present),
            "inp_path": _inp_path_from_payload(payload, metadata),
            "analysis_type": analysis_type,
            "material_model": str(metadata.get("material_model") or "linear_elastic"),
            "frame_count": frame_count,
            "has_frame_history": has_frame_history,
            "validation_branch": "nonlinear" if has_frame_history or analysis_type == "nonlinear_static" else "elastic",
            "supports_time_alignment": False,
            "supports_interpolated_validation": False,
            "analysis_stage": str(metadata.get("analysis_stage") or ("result" if has_c else "input_only")),
            "has_analysis_result": bool(has_c or is_shakedown),
            "analysis_input": bundle.get("analysis_input"),
            "analysis_input_history": bundle.get("analysis_input_history") or [],
            "inp_metadata": bundle.get("inp_metadata"),
            "metadata": {str(key): str(value) for key, value in metadata.items()},
        })

    return {
        "mat_count": len(details),
        "family": details[0]["family"] if details else "unknown",
        "is_validated": any(item["is_validated"] for item in details),
        "has_required_fields": all(item["has_required_fields"] for item in details),
        "missing_fields": sorted({miss for item in details for miss in item["missing_fields"]}),
        "details": details,
    }


def _candidate_odb_stems(mat_path: Path) -> list[str]:
    stems: list[str] = []

    def add(stem: str) -> None:
        text = str(stem).strip()
        if text and text not in stems:
            stems.append(text)

    add(mat_path.stem)
    try:
        import scipy.io as sio
        payload = _clean_mat_payload(sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False))
        metadata = _mat_metadata_dict(payload)
        inp_path = _inp_path_from_payload(payload, metadata)
        if inp_path:
            add(Path(inp_path).stem)
    except Exception:
        pass
    return stems


@router.post("/find-odb")
async def find_odb(req: FindOdbRequest) -> dict[str, Any]:
    mat_path = Path(req.mat_path)
    stems = _candidate_odb_stems(mat_path)
    search_dirs = [mat_path.parent, mat_path.parent / "abaqus", mat_path.parent.parent / "abaqus"]
    if req.model_path:
        model_root = Path(req.model_path)
        search_dirs.extend([model_root / "abaqus", model_root])

    for directory in search_dirs:
        for stem in stems:
            candidate = directory / f"{stem}.odb"
            if candidate.is_file():
                return {"found": True, "odb_path": str(candidate)}

    if req.model_path:
        model_root = Path(req.model_path)
        if model_root.is_dir():
            for stem in stems:
                for candidate in model_root.rglob(f"{stem}.odb"):
                    return {"found": True, "odb_path": str(candidate)}
    return {"found": False, "odb_path": None}
