"""WSL bridge — connectivity check and command execution."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_IN_WSL_PS1 = PROJECT_ROOT / "Config" / "run_in_wsl.ps1"
ENV_CFG = PROJECT_ROOT / "Config" / "env.cfg"


def _read_wsl_python_from_cfg() -> str:
    """Read wsl_python from Config/env.cfg, fall back to 'python3'."""
    try:
        for line in ENV_CFG.read_text(encoding="utf-8").splitlines():
            data = line.split("#", 1)[0].strip()
            if "=" in data:
                k, v = data.split("=", 1)
                if k.strip() == "wsl_python" and v.strip():
                    return v.strip()
    except Exception:
        pass
    return "python3"


def check_wsl_available() -> dict:
    """Detect whether WSL is installed on this machine.

    We check ``wsl --list`` (exit 0 when at least one distro is registered)
    and fall back to ``wsl --version`` so that WSL2 core-only installs are
    also detected.  Unlike "connectivity" checks this does **not** spin up a
    WSL session, so it is fast and reliable.
    """
    try:
        # Primary: wsl --list exits 0 if WSL is installed with ≥1 distro.
        result = subprocess.run(
            ["wsl", "--list", "--quiet"],
            capture_output=True,
            timeout=8,
        )
        if result.returncode == 0:
            return {"available": True, "detail": "WSL 已安装（distro 已注册）"}

        # Fallback: wsl --version works even if no distro is registered yet.
        ver_result = subprocess.run(
            ["wsl", "--version"],
            capture_output=True, text=True, timeout=8,
        )
        if ver_result.returncode == 0 or ver_result.stdout.strip():
            return {"available": True, "detail": ver_result.stdout.strip().splitlines()[0]}

        return {
            "available": False,
            "detail": "WSL 未安装或未启用（Windows 功能未开启）",
        }
    except FileNotFoundError:
        return {"available": False, "detail": "wsl.exe 未找到，WSL 未安装"}
    except subprocess.TimeoutExpired:
        return {"available": False, "detail": "WSL 检测超时"}


def detect_wsl_python() -> dict:
    """Detect JA Python interpreter version inside WSL."""
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "python3 --version 2>&1 || echo NOT_FOUND"],
            capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip()
        return {"version": version, "available": "NOT_FOUND" not in version}
    except Exception as exc:
        return {"version": "", "available": False, "error": str(exc)}


def _read_wsl_python_from_cfg() -> str:
    """Read *wsl_python* from Config/env.cfg; fall back to ``python3``."""
    cfg = PROJECT_ROOT / "Config" / "env.cfg"
    if not cfg.is_file():
        cfg = PROJECT_ROOT / "Config" / "env.template.cfg"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            data = line.split("#", 1)[0].strip()
            if data.startswith("wsl_python") and "=" in data:
                val = data.split("=", 1)[1].strip()
                if val:
                    return val
    return "python3"


def detect_jax_version() -> dict:
    """Detect JAX version by calling *wsl_python* directly via ``wsl -e``.

    Returns ``python_path`` so callers can auto-fill the config input.
    """
    wsl_python = _read_wsl_python_from_cfg()
    try:
        result = subprocess.run(
            ["wsl", "-e", wsl_python, "-c",
             "import jax; print(jax.__version__)"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "JAX_ENABLE_X64": "1"},
        )
        version = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        return {"version": version, "available": bool(version), "python_path": wsl_python}
    except FileNotFoundError:
        return {"version": "", "available": False, "python_path": wsl_python, "error": "wsl.exe 未找到"}
    except subprocess.TimeoutExpired:
        return {"version": "", "available": False, "python_path": wsl_python, "error": "JAX 检测超时（>30s）"}
    except Exception as exc:
        return {"version": "", "available": False, "python_path": wsl_python, "error": str(exc)}


def get_windows_python_info() -> dict:
    """Return info about the Windows Python running this server."""
    return {
        "version": sys.version.split()[0],
        "executable": sys.executable,
    }


def detect_windows_python_candidates() -> list[dict]:
    """Try common ways to locate a Windows Python interpreter.

    Returns a list of candidates, each with keys:
        executable (str), version (str), source (str).
    The list is ordered by preference (first = best).
    """
    import os

    candidates: list[dict] = []
    seen: set[str] = set()

    def _probe(cmd: list[str], source: str) -> None:
        try:
            result = subprocess.run(
                cmd + ["-c", "import sys; print(sys.version.split()[0]); print(sys.executable)"],
                capture_output=True, text=True, timeout=8,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                if len(lines) >= 2:
                    version, exe = lines[0].strip(), lines[1].strip()
                    if exe and exe.lower() not in seen:
                        from pathlib import Path as _Path
                        if _Path(exe).exists():
                            seen.add(exe.lower())
                            candidates.append({"executable": exe, "version": version, "source": source})
        except Exception:
            pass

    # 0. The currently running interpreter (highest priority — the server uses it)
    try:
        from pathlib import Path as _Path
        exe = sys.executable
        if exe and _Path(exe).exists() and exe.lower() not in seen:
            seen.add(exe.lower())
            candidates.append({
                "executable": exe,
                "version": sys.version.split()[0],
                "source": "running server",
            })
    except Exception:
        pass

    # 1. Python Launcher (most reliable on Windows 10/11)
    _probe(["py", "-3"], "py launcher")
    # 2. python in PATH
    _probe(["python"], "PATH (python)")
    # 3. python3 in PATH
    _probe(["python3"], "PATH (python3)")

    return candidates


def detect_wsl_python_candidates() -> list[dict]:
    """Scan WSL for Python interpreters that have JAX installed.

    Priority:
      1. Path already configured in env.cfg (verified first).
      2. Conda/venv paths discovered via ``find`` inside WSL.

    Returns a list (may be empty) of dicts with keys:
        path (str), py_version (str), jax_version (str), source (str).
    """
    import os

    configured = _read_wsl_python_from_cfg()
    candidates: list[dict] = []
    seen: set[str] = set()

    def _probe_wsl(wsl_path: str, source: str) -> "dict | None":
        """Return candidate dict if *wsl_path* inside WSL has JAX, else None."""
        try:
            result = subprocess.run(
                ["wsl", "-e", wsl_path, "-c",
                 "import sys, jax; print(sys.version.split()[0]); print(jax.__version__)"],
                capture_output=True, text=True, timeout=20,
                env={**os.environ, "JAX_ENABLE_X64": "1"},
            )
            if result.returncode == 0:
                lines = result.stdout.strip().splitlines()
                py_ver = lines[0].strip() if lines else ""
                jax_ver = lines[1].strip() if len(lines) >= 2 else ""
                if py_ver:
                    return {"path": wsl_path, "py_version": py_ver,
                            "jax_version": jax_ver, "source": source}
        except Exception:
            pass
        return None

    def _add(c: "dict | None") -> None:
        if c and c["path"] not in seen:
            seen.add(c["path"])
            candidates.append(c)

    # 1. Configured path (if set and not the generic fallback)
    if configured and configured != "python3":
        _add(_probe_wsl(configured, "env.cfg"))

    # 2. Discover conda/venv Pythons via WSL find (bounded depth)
    try:
        r = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "find /home /root /opt -maxdepth 8 \\( -type f -o -type l \\) -name python 2>/dev/null"
             " | grep -E '(envs|miniconda|anaconda|conda)' | sort -u | head -15"],
            capture_output=True, text=True, timeout=15,
        )
        for line in r.stdout.strip().splitlines():
            p = line.strip()
            if p and p not in seen:
                _add(_probe_wsl(p, "auto-scan"))
    except Exception:
        pass

    # 3. Fallback: try generic python3 / python on WSL PATH
    for fallback in ("python3", "python"):
        if fallback not in seen:
            _add(_probe_wsl(fallback, "WSL PATH"))

    def _candidate_rank(candidate: dict) -> tuple:
        path = candidate["path"]
        return (
            0 if candidate["source"] == "env.cfg" else 1,
            0 if "/envs/" in path or "/.venv/" in path or "/venv/" in path else 1,
            0 if any(token in path for token in ("miniconda", "anaconda", "conda")) else 1,
            path,
        )

    candidates.sort(key=_candidate_rank)
    return candidates


async def run_wsl_module_async(
    module: str,
    args: list[str],
    *,
    on_stdout: Optional[callable] = None,
    on_stderr: Optional[callable] = None,
) -> int:
    """Run a jaxmech module via ``wsl -e bash`` asynchronously.

    Bypasses the PowerShell wrapper to avoid output buffering, which would
    prevent real-time log streaming to the WebSocket.

    Returns the process exit code.  *on_stdout* / *on_stderr* are called with
    each line as it arrives.
    """
    import os

    wsl_python = _read_wsl_python_from_cfg()
    project_root_wsl = _win_to_wsl_path(str(PROJECT_ROOT))
    pythonpath = (
        f"{project_root_wsl}/external"
        f":{project_root_wsl}"
    )

    # Detect the real Linux HOME to avoid MPI/PETSc init failures.
    try:
        home_result = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "getent passwd $(id -un) 2>/dev/null | cut -d: -f6"],
            capture_output=True, text=True, timeout=5,
        )
        wsl_home = home_result.stdout.strip() or "/root"
    except Exception:
        wsl_home = "/root"

    # Build the args string for the module invocation.
    # Convert any Windows paths in args to WSL paths.
    wsl_args = [_win_to_wsl_path(a) if _looks_like_win_path(a) else a for a in args]
    args_str = " ".join(wsl_args)

    bash_cmd = (
        f"HOME={wsl_home} "
        f"PYTHONPATH={pythonpath} "
        f"JAX_ENABLE_X64=1 "
        f"PYTHONUNBUFFERED=1 "
        f"{wsl_python} -u -m {module} {args_str}"
    )

    cmd = ["wsl", "-e", "bash", "-c", bash_cmd]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
    )

    async def _stream(pipe, callback):
        while True:
            line = await pipe.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if callback:
                await callback(text)

    await asyncio.gather(
        _stream(proc.stdout, on_stdout),
        _stream(proc.stderr, on_stderr),
    )
    await proc.wait()
    return proc.returncode


def _win_to_wsl_path(path: str) -> str:
    """Convert a Windows path to a WSL /mnt/... path if applicable."""
    import re
    path = path.replace("\\", "/")
    m = re.match(r'^([A-Za-z]):/(.*)', path)
    if m:
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return path


def _looks_like_win_path(s: str) -> bool:
    import re
    return bool(re.match(r'^[A-Za-z]:[/\\]', s))
