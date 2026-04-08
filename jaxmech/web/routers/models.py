"""Models API - browse and inspect models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import numpy as np

from jaxmech.io.abaqus.inp_metadata import parse_inp_metadata_batch
from jaxmech.io.abaqus.inp_metadata_store import (
    decode_analysis_input_bundle,
    save_analysis_input_snapshot,
)
try:
    from jaxmech.modules.direct_methods.dca.inp import parse_direct_cyclic_inp
    _HAS_DCA = True
except ImportError:
    _HAS_DCA = False
from jaxmech.web.services.model_scanner import get_model_detail, scan_models

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
async def list_models():
    """List all available models from Examples/ and StoredModels/."""
    return scan_models()


@router.get("/{source}/{name}")
async def model_detail(source: str, name: str):
    """Get detailed status for one model."""
    info = get_model_detail(source, name)
    if info is None:
        raise HTTPException(404, f"Model not found: {source}/{name}")
    return info


class ScanMatsRequest(BaseModel):
    model_path: str  # Windows absolute path to the model root directory


class InpMetaRequest(BaseModel):
    paths: List[str]


class DcaInpMetaRequest(BaseModel):
    paths: List[str]


class PrepareIncAnalysisRequest(BaseModel):
    model_path: str
    inp_paths: List[str]
    analysis_input: Dict[str, Any]
    source: str = "web_inc_analysis"


@router.post("/scan-mats")
async def scan_mats(req: ScanMatsRequest) -> Dict[str, Any]:
    """
    Scan model root directory recursively for .mat files,
    grouped by subdirectory (e.g., inc_analysis/).
    Returns a flat list with relative paths.
    """
    model_dir = Path(req.model_path)
    if not model_dir.is_dir():
        raise HTTPException(404, f"Model directory not found: {req.model_path}")

    results: list[dict[str, Any]] = []
    for mat_path in sorted(model_dir.rglob("*.mat")):
        rel = mat_path.relative_to(model_dir)
        results.append({
            "abs_path": str(mat_path),
            "rel_path": str(rel),
            "name": mat_path.name,
            "subdir": str(rel.parent) if len(rel.parts) > 1 else "",
        })

    return {"model_path": str(model_dir), "mat_files": results}


class MatMetaRequest(BaseModel):
    paths: List[str]  # Windows absolute paths to .mat files


def _unwrap_scalar(value):
    cur = value
    while isinstance(cur, np.ndarray) and cur.size == 1:
        cur = cur.reshape(-1)[0]
    return cur


def _clean_mat_payload(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _unwrap_scalar(value)
        for key, value in raw.items()
        if not key.startswith("__")
    }


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


def _inp_path_from_payload(payload: dict[str, Any]) -> str:
    if "InpData" in payload:
        inp_data = _unwrap_scalar(payload["InpData"])
        if isinstance(inp_data, dict):
            inp_path = inp_data.get("inp_path")
            if inp_path is not None:
                return str(_unwrap_scalar(inp_path)).strip()
        elif hasattr(inp_data, "inp_path"):
            return str(_unwrap_scalar(getattr(inp_data, "inp_path"))).strip()
    metadata = _mat_metadata_dict(payload)
    for key in ("source_inp", "source_file"):
        value = metadata.get(key)
        if value is not None:
            text = str(_unwrap_scalar(value)).strip()
            if text:
                return text
    return ""


def _direct_cyclic_info_from_payload(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    dci = _struct_to_dict(payload.get("DirectCyclicInput"))
    solver_settings = _struct_to_dict(dci.get("solver_settings")) if dci else {}

    direct_branch = "dca" if dci or "DirectCyclicResult" in payload else ""
    if direct_branch:
        info["direct_method_branch"] = direct_branch
        info["source_inp"] = str(dci.get("source_inp") or metadata.get("source_inp") or metadata.get("source_file") or "").strip()
        info["step_name"] = str(dci.get("step_name") or "").strip()
        info["amplitude_name"] = str(dci.get("amplitude_name") or "").strip()
        info["abaqus_reference_mat"] = str(dci.get("abaqus_reference_mat") or metadata.get("abaqus_reference_mat") or "").strip()
        info["constitutive_backend"] = str(dci.get("constitutive_backend") or "").strip()
        info["n_harmonics"] = int(float(solver_settings.get("n_harmonics") or 0)) if solver_settings.get("n_harmonics") is not None else 0
        info["n_time_points"] = int(float(solver_settings.get("n_time_points") or 0)) if solver_settings.get("n_time_points") is not None else 0
        info["periodic_start_iteration"] = int(float(solver_settings.get("period_con_from_iteration") or 0)) if solver_settings.get("period_con_from_iteration") is not None else 0
    return info


def _infer_mat_analysis_info(payload: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    direct_info = _direct_cyclic_info_from_payload(payload, metadata)
    if direct_info.get("direct_method_branch") == "dca":
        frame_raw = payload.get("frame_u")
        frame_count = int(np.asarray(frame_raw).shape[0]) if frame_raw is not None else 0
        return {
            "analysis_type": "direct_cyclic",
            "material_model": str(metadata.get("material_model") or direct_info.get("constitutive_backend") or "j2_perfect_plastic"),
            "frame_count": frame_count,
            "has_frame_history": frame_count > 0,
            "validation_branch": "nonlinear",
            "supports_time_alignment": False,
            "supports_interpolated_validation": False,
            "supports_abaqus_mat_compare": True,
            "direct_method_branch": "dca",
        }

    frame_raw = payload.get("frame_u")
    frame_count = int(np.asarray(frame_raw).shape[0]) if frame_raw is not None else 0
    has_frame_history = frame_count > 0
    material_model = str(metadata.get("material_model") or "").strip() or (
        "j2_perfect_plastic" if "PlasticResult" in payload else "linear_elastic"
    )
    analysis_type = str(metadata.get("analysis_type") or "").strip() or (
        "nonlinear_static" if "PlasticResult" in payload or frame_count > 1 else "linear_static"
    )
    validation_branch = "nonlinear" if analysis_type == "nonlinear_static" or frame_count > 1 else "elastic"
    return {
        "analysis_type": analysis_type,
        "material_model": material_model,
        "frame_count": frame_count,
        "has_frame_history": has_frame_history,
        "validation_branch": validation_branch,
        "supports_time_alignment": validation_branch == "nonlinear",
        "supports_interpolated_validation": validation_branch == "nonlinear" and has_frame_history,
        "supports_abaqus_mat_compare": False,
        "direct_method_branch": "",
    }


@router.post("/inp-meta")
async def inp_meta(req: InpMetaRequest) -> Dict[str, Any]:
    """Parse INP files and return lightweight metadata for Web forms."""
    results = parse_inp_metadata_batch(req.paths)
    return {"count": len(results), "details": results}


@router.post("/dca-inp-meta")
async def dca_inp_meta(req: DcaInpMetaRequest) -> Dict[str, Any]:
    """Parse DCA-specific step metadata from one or more INP files for Web forms."""
    if not _HAS_DCA:
        raise HTTPException(501, "DCA module is not available in this installation.")
    results: list[dict[str, Any]] = []
    for raw_path in req.paths:
        inp_path = Path(raw_path)
        if not inp_path.is_file():
            raise HTTPException(404, f"INP file not found: {raw_path}")
        try:
            spec = parse_direct_cyclic_inp(inp_path)
            abaqus_reference = inp_path.with_suffix(".mat")
            inp_meta = parse_inp_metadata_batch([str(inp_path)])[0]
            materials = inp_meta.get("materials") or []
            results.append({
                "path": str(inp_path),
                "name": inp_path.name,
                "supported": True,
                "family": str(inp_meta.get("family") or ""),
                "parser_scope": str(inp_meta.get("parser_scope") or ""),
                "support_note": str(inp_meta.get("support_note") or ""),
                "supports_elastoplastic": bool(inp_meta.get("supports_elastoplastic", False)),
                "suggested_material_model": str(inp_meta.get("suggested_material_model") or ""),
                "materials": materials,
                "source_inp": str(spec.inp_path),
                "step_name": str(spec.step_name),
                "amplitude_name": str(spec.amplitude_name),
                "period": float(spec.period),
                "time_increment": float(spec.time_increment),
                "n_time_points": int(spec.n_time_points),
                "n_harmonics": int(spec.n_harmonics),
                "max_iterations": -1 if spec.max_iterations is None else int(spec.max_iterations),
                "load_surface_name": str(spec.load_surface_name),
                "load_type": str(spec.load_type),
                "load_magnitude": float(spec.load_magnitude),
                "abaqus_reference_mat": str(abaqus_reference) if abaqus_reference.is_file() else "",
                "time_grid_preview": np.asarray(spec.time_grid[: min(5, spec.time_grid.size)], dtype=np.float64).tolist(),
                "load_table_preview": np.asarray(spec.load_table[: min(5, spec.load_table.size)], dtype=np.float64).tolist(),
            })
        except Exception as exc:
            results.append({
                "path": str(inp_path),
                "name": inp_path.name,
                "supported": False,
                "error": str(exc),
            })
    return {"count": len(results), "details": results}


@router.post("/prepare-inc-analysis")
async def prepare_inc_analysis(req: PrepareIncAnalysisRequest) -> Dict[str, Any]:
    """Persist temporary INP parse metadata and editable input before inc_analysis runs."""
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
        if inp_meta.get("supported") is False:
            raise HTTPException(422, f"INP is not supported for Web inc_analysis: {inp_path.name}")

        result_mat_path = model_dir / "inc_analysis" / f"{inp_path.stem}.mat"
        status = save_analysis_input_snapshot(
            result_mat_path,
            inp_meta,
            req.analysis_input,
            source=req.source,
        )
        results.append(
            {
                "inp_path": str(inp_path),
                "inp_name": inp_path.name,
                "result_mat_path": status["result_mat_path"],
                "sidecar_mat_path": status["sidecar_mat_path"],
                "changed_fields": status["changed_fields"],
                "history_count": status["history_count"],
                "updated_result_mat": status["updated_result_mat"],
                "analysis_input": status["analysis_input"],
            }
        )

    return {
        "count": len(results),
        "workflow_dir": str(model_dir / "inc_analysis"),
        "sidecar_dir": str((model_dir / "inc_analysis" / "inputs")),
        "details": results,
    }


@router.post("/mat-meta")
async def mat_meta(req: MatMetaRequest) -> Dict[str, Any]:
    """
    Read one or more .mat files and check for required shakedown fields.
    Returns: family, is_validated, has_required_fields, missing_fields, mat_count, details.
    """
    try:
        import scipy.io  # type: ignore
    except ImportError:
        raise HTTPException(500, "scipy not available in Windows Python environment")

    def _str_field(mat: dict, key: str) -> str:
        """Safely extract a string scalar from a mat field."""
        raw = mat.get(key, None)
        if raw is None:
            return ""
        if hasattr(raw, "item"):
            return str(raw.item()).strip()
        if hasattr(raw, "tolist"):
            return str(raw.tolist()).strip()
        return str(raw).strip()

    def _extract_family(mat: dict) -> str:
        """
        Try metadata.family first (preferred per repo convention),
        then fall back to top-level 'family' field.
        """
        # Try metadata struct
        metadata = mat.get("metadata", None)
        if metadata is not None:
            try:
                fam = getattr(metadata, "family", None)
                if fam is not None:
                    if hasattr(fam, "item"):
                        return str(fam.item()).strip()
                    return str(fam).strip()
            except Exception:
                pass
        # Fall back to top-level 'family'
        return _str_field(mat, "family") or "unknown"

    def _check_validated(mat: dict) -> tuple[bool, bool]:
        """
        Check whether the MAT file contains any validated-related fields.
        Looks for: 'validated_with_ODB' (value==1), 'validated', 'validation_passed',
        'odb_validation', 'is_validated'.
        """
        present = {k for k in mat.keys() if not k.startswith("__")}
        # Primary key written by JAX validation pipeline
        if "validated_with_ODB" in present:
            try:
                import numpy as _np
                val = _np.asarray(mat["validated_with_ODB"], dtype=_np.int32).reshape(-1)[0]
                return int(val) == 1, True
            except Exception:
                return bool(mat["validated_with_ODB"]), True
        # Legacy / alternative keys
        validated_keys = {"validated", "validation_passed", "odb_validation", "is_validated"}
        for key in validated_keys:
            if key in present:
                try:
                    import numpy as _np
                    val = _np.asarray(mat[key]).reshape(-1)[0]
                    return bool(val), True
                except Exception:
                    return bool(mat[key]), True
        return False, False


    results: list[dict[str, Any]] = []
    for raw_path in req.paths:
        p = Path(raw_path)
        if not p.is_file():
            raise HTTPException(404, f"MAT file not found: {raw_path}")
        try:
            mat = scipy.io.loadmat(str(p), squeeze_me=True, struct_as_record=False)
        except Exception as e:
            raise HTTPException(422, f"Failed to read {p.name}: {e}")

        payload = _clean_mat_payload(mat)
        metadata = _mat_metadata_dict(payload)
        bundle = decode_analysis_input_bundle(payload)
        analysis_input = bundle.get("analysis_input")
        analysis_input_history = bundle.get("analysis_input_history") or []
        inp_metadata = bundle.get("inp_metadata")

        family = _extract_family(mat)
        is_validated, has_validation_flag = _check_validated(mat)
        analysis_info = _infer_mat_analysis_info(payload, metadata)
        inp_path = _inp_path_from_payload(payload)
        direct_info = _direct_cyclic_info_from_payload(payload, metadata)

        # Check required fields (C_sparse OR C_gen must be present)
        present = {k for k in mat.keys() if not k.startswith("__")}
        has_c = "C_sparse" in present or "C_gen" in present
        missing = []
        if not has_c:
            missing = ["C_sparse / C_gen"]
        analysis_stage = str(metadata.get("analysis_stage") or ("result" if has_c else "input_only")).strip()

        results.append({
            "path": raw_path,
            "name": p.name,
            "family": family,
            "is_validated": is_validated,
            "has_validation_flag": has_validation_flag,
            "has_required_fields": len(missing) == 0,
            "missing_fields": missing,
            "all_fields": sorted(present),
            "inp_path": inp_path,
            "analysis_type": analysis_info["analysis_type"],
            "material_model": analysis_info["material_model"],
            "frame_count": analysis_info["frame_count"],
            "has_frame_history": analysis_info["has_frame_history"],
            "validation_branch": analysis_info["validation_branch"],
            "supports_time_alignment": analysis_info["supports_time_alignment"],
            "supports_interpolated_validation": analysis_info["supports_interpolated_validation"],
            "supports_abaqus_mat_compare": analysis_info["supports_abaqus_mat_compare"],
            "direct_method_branch": analysis_info["direct_method_branch"],
            "analysis_stage": analysis_stage,
            "has_analysis_result": has_c,
            "analysis_input": analysis_input,
            "analysis_input_history": analysis_input_history,
            "inp_metadata": inp_metadata,
            "direct_method_info": direct_info,
            "metadata": {str(key): str(value) for key, value in metadata.items()},
        })

    # Aggregate summary
    overall_family = results[0]["family"] if results else "unknown"
    all_valid = all(r["has_required_fields"] for r in results)
    all_missing = sorted({f for r in results for f in r["missing_fields"]})
    any_validated = any(r["is_validated"] for r in results)

    return {
        "mat_count": len(results),
        "family": overall_family,
        "is_validated": any_validated,
        "has_required_fields": all_valid,
        "missing_fields": all_missing,
        "details": results,
    }


# ---------------------------------------------------------------------------
# find-odb: auto-locate same-named ODB for a given MAT file
# ---------------------------------------------------------------------------

class FindOdbRequest(BaseModel):
    mat_path: str   # Windows absolute path to the .mat file
    model_path: str = ""  # Windows absolute path to the model root (optional, for broader search)


@router.post("/find-odb")
async def find_odb(req: FindOdbRequest) -> Dict[str, Any]:
    """
    Given a .mat file path, search for a same-named .odb file.

    Search order:
      1. Same directory as the .mat file
      2. <mat_dir>/abaqus/
      3. <mat_dir>/../abaqus/  (one level up)
      4. <model_root>/abaqus/  (if model_path provided)
      5. Recursive search under model_root (slowest, only if model_path provided)
    """
    mat_p = Path(req.mat_path)
    stem = mat_p.stem  # filename without extension
    odb_name = stem + ".odb"

    search_dirs: list[Path] = [mat_p.parent]
    search_dirs.append(mat_p.parent / "abaqus")
    search_dirs.append(mat_p.parent.parent / "abaqus")
    if req.model_path:
        model_root = Path(req.model_path)
        search_dirs.append(model_root / "abaqus")
        search_dirs.append(model_root)

    # Exact-match search in listed directories
    for d in search_dirs:
        candidate = d / odb_name
        if candidate.is_file():
            return {"found": True, "odb_path": str(candidate)}

    # Recursive fallback under model_root
    if req.model_path:
        for candidate in Path(req.model_path).rglob(odb_name):
            return {"found": True, "odb_path": str(candidate)}

    return {"found": False, "odb_path": None}
