"""Status and environment check API for the demo subset."""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from jaxmech.web.services.config_manager import write_env_cfg
from jaxmech.web.services.wsl_bridge import (
    check_wsl_available,
    detect_jax_version,
    detect_windows_python_candidates,
    detect_wsl_python,
    detect_wsl_python_candidates,
    get_windows_python_info,
)


router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("")
async def get_status() -> dict[str, Any]:
    wsl, wsl_py, jax, libs = await asyncio.gather(
        asyncio.to_thread(check_wsl_available),
        asyncio.to_thread(detect_wsl_python),
        asyncio.to_thread(detect_jax_version),
        asyncio.to_thread(_check_python_libraries),
    )
    return {
        "wsl": wsl,
        "wsl_python": wsl_py,
        "jax": jax,
        "windows_python": get_windows_python_info(),
        "python_libraries": libs,
    }


@router.get("/quick")
async def get_quick_status() -> dict[str, Any]:
    return {
        "wsl": await asyncio.to_thread(check_wsl_available),
        "windows_python": get_windows_python_info(),
    }


def _check_python_libraries() -> dict[str, Any]:
    required = {
        "numpy": "Numerical arrays",
        "scipy": "MAT file IO",
        "fastapi": "Web API",
        "uvicorn": "ASGI server",
        "pydantic": "API validation",
        "websockets": "Live task logs",
        "pyvista": "MAT visualization rendering",
        "vtk": "PyVista rendering backend",
        "PIL": "Image export",
    }
    libraries: dict[str, Any] = {}
    missing: list[str] = []
    for name, required_for in required.items():
        try:
            module = __import__(name)
            libraries[name] = {
                "available": True,
                "required_for": required_for,
                "version": getattr(module, "__version__", getattr(module, "VERSION", "unknown")),
            }
        except ImportError:
            missing.append(name)
            package = "Pillow" if name == "PIL" else name
            libraries[name] = {
                "available": False,
                "required_for": required_for,
                "version": None,
                "install_cmd": [sys.executable, "-m", "pip", "install", package],
            }
    return {
        "libraries": libraries,
        "missing_count": len(missing),
        "missing_libraries": missing,
        "all_available": not missing,
    }


def _parse_gurobi_version(path: Path) -> tuple[int, ...]:
    match = re.search(r"gurobi(\d+)", path.name.lower())
    if not match:
        return (10**9,)
    digits = match.group(1)
    if len(digits) >= 3:
        return (int(digits[:-2] or 0), int(digits[-2]), int(digits[-1]))
    if len(digits) == 2:
        return (int(digits[0]), int(digits[1]), 0)
    return (int(digits), 0, 0)


def _detect_gurobi_root() -> dict[str, Any]:
    base = Path(r"C:\\")
    candidates: list[Path] = []
    try:
        for path in base.iterdir():
            if path.is_dir() and re.fullmatch(r"gurobi\d+", path.name.lower()):
                candidates.append(path)
    except Exception as exc:
        return {"found": False, "root": "", "source": "", "error": str(exc)}

    valid = [path for path in candidates if (path / "win64").is_dir()]
    if not valid:
        valid = candidates
    if not valid:
        return {"found": False, "root": "", "source": r"C:\gurobi*"}
    ordered = sorted(valid, key=lambda p: (_parse_gurobi_version(p), p.name.lower()))
    best = ordered[0]
    return {
        "found": True,
        "root": str(best),
        "source": r"C:\gurobi*",
        "version": ".".join(str(v) for v in _parse_gurobi_version(best)),
        "candidates": [str(path) for path in ordered],
    }


def _detect_abaqus_cmd() -> dict[str, Any]:
    try:
        result = subprocess.run(["where.exe", "abaqus"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            found_path = result.stdout.strip().splitlines()[0].strip()
            if found_path:
                return {"found": True, "abaqus_cmd": "abaqus", "source": f"PATH ({found_path})"}
    except Exception:
        pass

    common = Path(r"C:\SIMULIA\Commands\abaqus.bat")
    if common.is_file():
        return {"found": True, "abaqus_cmd": str(common), "source": r"C:\SIMULIA\Commands"}

    simulia = Path(r"C:\SIMULIA")
    if simulia.is_dir():
        for candidate in simulia.rglob("abaqus.bat"):
            return {"found": True, "abaqus_cmd": str(candidate), "source": str(candidate.parent)}
        for candidate in simulia.rglob("abaqus.cmd"):
            return {"found": True, "abaqus_cmd": str(candidate), "source": str(candidate.parent)}

    return {"found": False, "abaqus_cmd": None, "source": ""}


@router.get("/detect-windows-python")
async def detect_windows_python(save: bool = False) -> dict[str, Any]:
    candidates = await asyncio.to_thread(detect_windows_python_candidates)
    if not candidates:
        return {"found": False, "candidates": [], "saved": False}
    best = candidates[0]
    if save:
        write_env_cfg({"windows_python_exe": best["executable"]})
    return {"found": True, "candidates": candidates, "best": best, "saved": bool(save)}


@router.get("/detect-gurobi-root")
async def detect_gurobi_root() -> dict[str, Any]:
    return await asyncio.to_thread(_detect_gurobi_root)


@router.get("/detect-abaqus")
async def detect_abaqus() -> dict[str, Any]:
    return await asyncio.to_thread(_detect_abaqus_cmd)


@router.get("/detect-wsl-python")
async def detect_wsl_python_endpoint(save: bool = False) -> dict[str, Any]:
    candidates = await asyncio.to_thread(detect_wsl_python_candidates)
    if not candidates:
        return {"found": False, "candidates": [], "saved": False}
    best = candidates[0]
    if save:
        write_env_cfg({"wsl_python": best["path"]})
    return {"found": True, "candidates": candidates, "best": best, "saved": bool(save)}


@router.post("/install-library/{library_name}")
async def install_library(library_name: str) -> dict[str, Any]:
    lib_check = _check_python_libraries()
    if library_name not in lib_check["libraries"]:
        return {"success": False, "error": f"Unknown library: {library_name}"}
    lib_info = lib_check["libraries"][library_name]
    if lib_info["available"]:
        return {"success": True, "message": f"{library_name} is already installed", "already_installed": True}
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            lib_info["install_cmd"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    return {
        "success": result.returncode == 0,
        "message": f"{library_name} installed" if result.returncode == 0 else "Installation failed",
        "output": result.stdout,
        "stderr": result.stderr,
    }


@router.post("/install-missing-libraries")
async def install_missing_libraries() -> dict[str, Any]:
    lib_check = _check_python_libraries()
    results = []
    for name in lib_check["missing_libraries"]:
        results.append({"library": name, **await install_library(name)})
    return {
        "success": all(item.get("success") for item in results),
        "message": "All supported libraries are already installed" if not results else "Supported library installation finished",
        "results": results,
        "success_count": sum(1 for item in results if item.get("success")),
        "total_count": len(results),
    }


@router.get("/modules")
async def get_module_availability() -> dict[str, bool]:
    from importlib.util import find_spec

    probes = {
        "elastic": "jaxmech.modules.inc_analysis.elastic",
        "shakedown": "jaxmech.modules.shakedown",
        "visualization": "jaxmech.modules.visualization",
        "optimize_cvxpy": "jaxmech.optimize.cvxpy",
    }
    availability = {key: find_spec(path) is not None for key, path in probes.items()}
    availability.update(
        {
            "validation": False,
            "direct_methods": False,
            "direct_methods_steady_state": False,
            "direct_methods_shakedown": False,
            "shakedown_shell": False,
            "shakedown_gurobi": False,
            "plastic": False,
            "topopt": False,
        }
    )
    return availability
