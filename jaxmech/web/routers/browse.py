"""File/folder browser API — opens a native Windows dialog via PowerShell.

Supported ``field_type`` values and their dialog behaviour:

  wsl_python        → OpenFileDialog, "All files", start at \\\\wsl$\\Ubuntu\\home\\
  windows_python    → OpenFileDialog, filter "python.exe", start at C:\\
  win_folder        → FolderBrowserDialog, start at C:\\
  wsl_folder        → FolderBrowserDialog, start at \\\\wsl$\\Ubuntu\\
  file              → OpenFileDialog, "All files", start at C:\\   (generic)
  folder            → FolderBrowserDialog, start at C:\\           (generic)
  mat_file          → OpenFileDialog, filter ".mat", start at initial_dir (model dir)
"""

from __future__ import annotations

import subprocess
from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/browse", tags=["browse"])


# WSL distro root as visible from Windows Explorer
_WSL_ROOT = r"\\wsl$\Ubuntu"
_WSL_HOME = rf"{_WSL_ROOT}\home"


class BrowseRequest(BaseModel):
    field_type: str = "folder"          # see module docstring
    title: Optional[str] = None
    initial_dir: Optional[str] = None


class BrowseResponse(BaseModel):
    path: Optional[str] = None          # None = user cancelled


# ---------------------------------------------------------------------------
# PowerShell helpers
# ---------------------------------------------------------------------------

def _ps_file_dialog(title: str, initial_dir: str, file_filter: str) -> Optional[str]:
    """OpenFileDialog → selected file path or None."""
    init = initial_dir.replace("'", "")
    filt = file_filter.replace("'", "")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.OpenFileDialog;"
        f"$d.Title = '{title}';"
        f"$d.InitialDirectory = '{init}';"
        f"$d.Filter = '{filt}';"
        "$d.Multiselect = $false;"
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.FileName }"
    )
    r = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120,
    )
    out = r.stdout.strip()
    return out if out else None


def _ps_folder_dialog(title: str, initial_dir: str) -> Optional[str]:
    """FolderBrowserDialog → selected folder path or None."""
    init = initial_dir.replace("'", "")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        f"$d.Description = '{title}';"
        f"$d.SelectedPath = '{init}';"
        "$d.ShowNewFolderButton = $true;"
        "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
    )
    r = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=120,
    )
    out = r.stdout.strip()
    return out if out else None


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

_FIELD_CONFIG: dict[str, dict] = {
    "wsl_python": {
        "kind": "file",
        "title": "选择 WSL Python 可执行文件",
        "initial": _WSL_HOME,
        "filter": "所有文件 (*.*)|*.*",   # Linux 可执行文件无扩展名，必须显示全部文件
    },
    "windows_python": {
        "kind": "file",
        "title": "选择 Windows Python 可执行文件",
        "initial": r"C:\\",
        "filter": "Python 可执行文件 (python.exe)|python.exe|所有文件 (*.*)|*.*",
    },
    "win_folder": {
        "kind": "folder",
        "title": "选择 Windows 目录",
        "initial": r"C:\\",
    },
    "wsl_folder": {
        "kind": "folder",
        "title": "选择 WSL 目录",
        "initial": _WSL_HOME,
    },
    "file": {
        "kind": "file",
        "title": "选择文件",
        "initial": r"C:\\",
        "filter": "所有文件 (*.*)|*.*",
    },
    "folder": {
        "kind": "folder",
        "title": "选择目录",
        "initial": r"C:\\",
    },
    "mat_file": {
        "kind": "file",
        "title": "选择 MAT 文件",
        "initial": r"C:\\",
        "filter": "MAT 文件 (*.mat)|*.mat|所有文件 (*.*)|*.*",
    },
    "odb_file": {
        "kind": "file",
        "title": "选择 ODB 文件",
        "initial": r"C:\\",
        "filter": "ODB 文件 (*.odb)|*.odb|所有文件 (*.*)|*.*",
    },
}



@router.post("", response_model=BrowseResponse)
async def browse(req: BrowseRequest) -> BrowseResponse:
    """Open a native Windows file/folder picker and return the selected path."""
    import asyncio

    cfg = _FIELD_CONFIG.get(req.field_type, _FIELD_CONFIG["folder"])
    title = req.title or cfg["title"]
    initial = req.initial_dir or cfg["initial"]

    if cfg["kind"] == "file":
        file_filter = cfg.get("filter", "所有文件 (*.*)|*.*")
        path = await asyncio.to_thread(_ps_file_dialog, title, initial, file_filter)
    else:
        path = await asyncio.to_thread(_ps_folder_dialog, title, initial)

    return BrowseResponse(path=path)
