"""Status & environment check API."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter

from jaxmech.web.services.wsl_bridge import (
    check_wsl_available,
    detect_jax_version,
    detect_windows_python_candidates,
    detect_wsl_python,
    detect_wsl_python_candidates,
    get_windows_python_info,
)
from jaxmech.web.services.config_manager import write_env_cfg

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("")
async def get_status():
    """Return combined environment status.

    All checks involve subprocess calls, so they run in a thread pool to
    avoid blocking the async event loop.
    """
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
async def get_quick_status():
    """Lightweight check (no JAX probe) for dashboard refresh."""
    wsl = await asyncio.to_thread(check_wsl_available)
    return {
        "wsl": wsl,
        "windows_python": get_windows_python_info(),
    }


def _check_python_libraries() -> Dict[str, Any]:
    """
    Check availability of required Python libraries in the Windows Python environment.
    
    Returns dict with library status and installation commands.
    """
    required_libs = {
        'websockets': {
            'description': 'WebSocket支持库',
            'required_for': '实时日志显示',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'websockets']
        },
        'scipy': {
            'description': 'SciPy科学计算库',  
            'required_for': '安定分析MAT文件读取',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'scipy']
        },
        'numpy': {
            'description': 'NumPy数值计算库',
            'required_for': '数值计算支持',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'numpy']
        },
        'fastapi': {
            'description': 'FastAPI Web框架',
            'required_for': 'Web服务运行',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'fastapi']
        },
        'uvicorn': {
            'description': 'Uvicorn ASGI服务器',
            'required_for': 'Web服务运行',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'uvicorn']
        },
        'pydantic': {
            'description': 'Pydantic数据验证',
            'required_for': 'API数据验证',
            'install_cmd': [sys.executable, '-m', 'pip', 'install', 'pydantic']
        }
    }
    
    results = {}
    missing_libraries = []
    
    for lib_name, lib_info in required_libs.items():
        try:
            __import__(lib_name)
            results[lib_name] = {
                'available': True,
                'description': lib_info['description'],
                'required_for': lib_info['required_for'],
                'version': _get_library_version(lib_name)
            }
        except ImportError:
            results[lib_name] = {
                'available': False,
                'description': lib_info['description'],
                'required_for': lib_info['required_for'],
                'install_cmd': lib_info['install_cmd'],
                'version': None
            }
            missing_libraries.append(lib_name)
    
    return {
        'libraries': results,
        'missing_count': len(missing_libraries),
        'missing_libraries': missing_libraries,
        'all_available': len(missing_libraries) == 0
    }


def _get_library_version(library_name: str) -> str:
    """Get version of an installed library."""
    try:
        if library_name == 'websockets':
            import websockets
            return getattr(websockets, '__version__', 'unknown')
        elif library_name == 'scipy':
            import scipy
            return getattr(scipy, '__version__', 'unknown')
        elif library_name == 'numpy':
            import numpy
            return getattr(numpy, '__version__', 'unknown')
        elif library_name == 'fastapi':
            import fastapi
            return getattr(fastapi, '__version__', 'unknown')
        elif library_name == 'uvicorn':
            import uvicorn
            return getattr(uvicorn, '__version__', 'unknown')
        elif library_name == 'pydantic':
            import pydantic
            return getattr(pydantic, 'VERSION', 'unknown')
        else:
            return 'unknown'
    except Exception:
        return 'unknown'


def _detect_abaqus_cmd() -> dict:
    """
    Try to locate the ABAQUS command on Windows.

    Search order:
      1. `where.exe abaqus`  — checks system PATH
      2. C:\\SIMULIA\\Commands\\abaqus.bat  (common install location)
      3. Any abaqus.bat / abaqus.cmd under C:\\SIMULIA (first match)

    Returns {'found': bool, 'abaqus_cmd': str|None, 'source': str}.
    """
    # 1. Check PATH
    try:
        r = subprocess.run(
            ["where.exe", "abaqus"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            found_path = r.stdout.strip().splitlines()[0].strip()
            if found_path:
                return {"found": True, "abaqus_cmd": "abaqus", "source": f"PATH ({found_path})"}
    except Exception:
        pass

    # 2. Common install: C:\SIMULIA\Commands\abaqus.bat
    common = Path(r"C:\SIMULIA\Commands\abaqus.bat")
    if common.is_file():
        return {"found": True, "abaqus_cmd": str(common), "source": r"C:\SIMULIA\Commands"}

    # 3. Broader search under C:\SIMULIA
    simulia = Path(r"C:\SIMULIA")
    if simulia.is_dir():
        for candidate in simulia.rglob("abaqus.bat"):
            return {"found": True, "abaqus_cmd": str(candidate), "source": str(candidate.parent)}
        for candidate in simulia.rglob("abaqus.cmd"):
            return {"found": True, "abaqus_cmd": str(candidate), "source": str(candidate.parent)}

    return {"found": False, "abaqus_cmd": None, "source": ""}


@router.get("/detect-abaqus")
async def detect_abaqus():
    """Auto-detect the Windows ABAQUS executable command."""
    return await asyncio.to_thread(_detect_abaqus_cmd)


@router.get("/detect-windows-python")
async def detect_windows_python(save: bool = False):
    """Auto-detect available Windows Python interpreters.

    When *save=true*, writes the first detected candidate to Config/env.cfg
    and returns it as the primary result.
    """
    candidates = await asyncio.to_thread(detect_windows_python_candidates)
    if not candidates:
        return {"found": False, "candidates": [], "saved": False}

    best = candidates[0]
    saved = False
    if save:
        write_env_cfg({"windows_python_exe": best["executable"]})
        saved = True

    return {
        "found": True,
        "candidates": candidates,
        "best": best,
        "saved": saved,
    }


@router.get("/detect-wsl-python")
async def detect_wsl_python_endpoint(save: bool = False):
    """Scan WSL for Python interpreters with JAX installed.

    When *save=true*, writes the best candidate to Config/env.cfg.
    """
    candidates = await asyncio.to_thread(detect_wsl_python_candidates)
    if not candidates:
        return {"found": False, "candidates": [], "saved": False}

    best = candidates[0]
    saved = False
    if save:
        write_env_cfg({"wsl_python": best["path"]})
        saved = True

    return {
        "found": True,
        "candidates": candidates,
        "best": best,
        "saved": saved,
    }


@router.post("/install-library/{library_name}")
async def install_library(library_name: str):
    """Install a specific Python library."""
    # Get library info from our requirements
    lib_check = _check_python_libraries()
    if library_name not in lib_check['libraries']:
        return {"success": False, "error": f"未知的库名称: {library_name}"}
    
    lib_info = lib_check['libraries'][library_name]
    if lib_info['available']:
        return {"success": True, "message": f"{library_name} 已经安装", "already_installed": True}
    
    install_cmd = lib_info['install_cmd']
    
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            install_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "message": f"{library_name} 安装成功",
                "output": result.stdout,
                "already_installed": False
            }
        else:
            return {
                "success": False,
                "error": f"安装失败: {result.stderr or result.stdout}",
                "output": result.stdout,
                "stderr": result.stderr
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "安装超时（5分钟）"}
    except Exception as e:
        return {"success": False, "error": f"安装出错: {str(e)}"}


# ── Module availability ──────────────────────────────────────────────

_MODULE_PROBES = {
    "elastic": "jaxmech.modules.inc_analysis.elastic",
    "shakedown": "jaxmech.modules.shakedown",
    "direct_methods": "jaxmech.modules.direct_methods.dca",
    "validation": "jaxmech.modules.validation",
    "validation_nonlinear": "jaxmech.modules.validation.nonlinear",
    "plastic": "jaxmech.modules.inc_analysis.plastic",
    "optimize": "jaxmech.optimize",
}

# Demo: these modules are explicitly unavailable regardless of sys.path
_DEMO_LOCKED = {"direct_methods", "validation", "validation_nonlinear", "plastic"}


def _probe_modules() -> Dict[str, bool]:
    """Check which optional analysis modules are importable."""
    from importlib.util import find_spec

    result = {}
    for key, mod_path in _MODULE_PROBES.items():
        if key in _DEMO_LOCKED:
            result[key] = False
        else:
            result[key] = find_spec(mod_path) is not None
    return result


@router.get("/modules")
async def get_module_availability():
    """Return a dict indicating which analysis modules are installed."""
    return await asyncio.to_thread(_probe_modules)


@router.post("/install-missing-libraries")  
async def install_missing_libraries():
    """Install all missing Python libraries in one go."""
    lib_check = _check_python_libraries()
    
    if lib_check['all_available']:
        return {"success": True, "message": "所有必需库都已安装", "results": []}
    
    missing_libs = lib_check['missing_libraries']
    results = []
    
    for lib_name in missing_libs:
        lib_info = lib_check['libraries'][lib_name]
        install_cmd = lib_info['install_cmd']
        
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                install_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                results.append({
                    "library": lib_name,
                    "success": True,
                    "message": f"{lib_name} 安装成功"
                })
            else:
                results.append({
                    "library": lib_name,
                    "success": False,
                    "error": f"安装失败: {result.stderr or result.stdout}"
                })
        except Exception as e:
            results.append({
                "library": lib_name,
                "success": False,
                "error": f"安装出错: {str(e)}"
            })
    
    success_count = len([r for r in results if r['success']])
    total_count = len(results)
    
    return {
        "success": success_count == total_count,
        "message": f"安装完成：{success_count}/{total_count} 个库安装成功",
        "results": results,
        "success_count": success_count,
        "total_count": total_count
    }
