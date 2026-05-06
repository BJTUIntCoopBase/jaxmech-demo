"""ABAQUS ODB extraction utilities used by jaxmech validation workflows."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from jaxmech.env.env import PROJECT_ROOT
from jaxmech.env.model_paths import win_to_wsl_path


def _to_windows_path_from_wsl(path_obj: str | Path) -> str:
    path = str(path_obj)
    if path.startswith("/mnt/") and len(path) > 6:
        drive = path[5].upper()
        tail = path[7:].replace("/", "\\")
        return f"{drive}:\\{tail}"
    return path.replace("/", "\\")


def _quote_win_cmd_arg(value: str | Path) -> str:
    text = str(value)
    return '"' + text.replace('"', '""') + '"'


def _format_win_command_name(value: str | Path) -> str:
    text = str(value)
    if any(ch.isspace() for ch in text) or "\\" in text or "/" in text:
        return _quote_win_cmd_arg(text)
    return text


def _run_windows_batch_script(script_path: Path, lines: list[str]) -> subprocess.CompletedProcess:
    script_path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    try:
        cmd = f'cmd.exe /d /c {_quote_win_cmd_arg(_to_windows_path_from_wsl(script_path))}'
        return subprocess.run(cmd, shell=True)
    finally:
        try:
            script_path.unlink()
        except FileNotFoundError:
            pass


def _verify_wsl_accessible(path: str | Path, *, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            "Make sure the Windows drive is mounted and the file exists."
        )


def extract_solid_odb(
    *,
    win_job_dir: str,
    odb_name: str,
    abaqus_cmd: str = "abaqus",
    linux_out_dir: str | None = None,
    skip_initial: bool = False,
    step: str | None = None,
    fixed_dof1_set=None,
    fixed_dof2_set=None,
    fixed_dof3_set=None,
) -> str:
    """Extract one solid/surface ABAQUS ODB into a MAT container."""
    win_job_dir = str(Path(win_job_dir))
    linux_out_dir = str(Path(linux_out_dir or win_job_dir))
    odb_linux = Path(win_job_dir) / odb_name
    exporter_src = PROJECT_ROOT / "jaxmech" / "io" / "abaqus" / "export_unified_odb.py"

    _verify_wsl_accessible(win_job_dir, label="Windows job directory")
    _verify_wsl_accessible(odb_linux, label="ODB file")
    _verify_wsl_accessible(exporter_src, label="ODB exporter script")

    exporter_dst = Path(win_job_dir) / exporter_src.name
    shutil.copy2(exporter_src, exporter_dst)

    odb_stem = Path(odb_name).stem
    mat_name = f"{odb_stem}.mat"
    mat_linux = Path(win_job_dir) / mat_name

    win_job = _to_windows_path_from_wsl(win_job_dir)
    win_odb = _to_windows_path_from_wsl(odb_linux)
    win_script = _to_windows_path_from_wsl(exporter_dst)
    win_prefix = _to_windows_path_from_wsl(Path(win_job_dir) / odb_stem)

    extra_flags = ""
    if skip_initial:
        extra_flags += " --skip-initial"
    if step:
        extra_flags += f" --step {step}"
    if fixed_dof1_set:
        extra_flags += f" --fixed-dof1-set {fixed_dof1_set}"
    if fixed_dof2_set:
        extra_flags += f" --fixed-dof2-set {fixed_dof2_set}"
    if fixed_dof3_set:
        extra_flags += f" --fixed-dof3-set {fixed_dof3_set}"

    runner_path = Path(win_job_dir) / "_jaxmech_extract_solid_odb.cmd"
    result = _run_windows_batch_script(
        runner_path,
        [
            "@echo off",
            f'cd /d {_quote_win_cmd_arg(win_job)} || exit /b 1',
            (
                f'call {_format_win_command_name(abaqus_cmd)} '
                f'python {_quote_win_cmd_arg(win_script)} '
                f'--odb {_quote_win_cmd_arg(win_odb)} '
                f'--out-prefix {_quote_win_cmd_arg(win_prefix)}'
                f'{extra_flags}'
            ),
            "exit /b %ERRORLEVEL%",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ABAQUS Python extraction failed for {odb_name} "
            f"(exit code {result.returncode})."
        )
    if not mat_linux.is_file():
        raise FileNotFoundError(f"Expected extracted MAT not found: {mat_linux}")

    out_dir = Path(linux_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / mat_name
    if mat_linux.resolve() != dst.resolve():
        shutil.copy2(mat_linux, dst)
    return str(dst)


def print_solid_extraction_commands(config: dict) -> None:
    """Print Windows-side commands for solid elastic ODB extraction."""
    model_name = str(config["model_name"])
    abaqus_cmd = str(config.get("windows_abaqus_cmd", "abaqus")).strip() or "abaqus"
    exporter_src = PROJECT_ROOT / "jaxmech" / "io" / "abaqus" / "export_unified_odb.py"
    exporter_win = _to_windows_path_from_wsl(exporter_src)
    win_root_wsl = win_to_wsl_path(config["win_stored_root"])

    for case_name in config["load_cases"]:
        odb_candidates = [
            win_root_wsl / model_name / "abaqus" / f"{case_name}.odb",
            win_root_wsl / model_name / f"{case_name}.odb",
            win_root_wsl / model_name / case_name / f"{case_name}.odb",
        ]
        odb_wsl = next((candidate for candidate in odb_candidates if candidate.is_file()), None)
        if odb_wsl is None:
            print(f"  WARNING: ODB not found for {case_name}")
            continue

        odb_dir_win = _to_windows_path_from_wsl(odb_wsl.parent)
        odb_win = _to_windows_path_from_wsl(odb_wsl)
        prefix_win = _to_windows_path_from_wsl(odb_wsl.with_suffix(""))
        print(f"  # {case_name}:")
        print(
            f"  cmd /c \"cd /d {odb_dir_win} && "
            f"{abaqus_cmd} python {exporter_win} --odb {odb_win} --out-prefix {prefix_win}\""
        )


def resolve_shell_export_path(
    odb_path: str | Path,
    *,
    out_path: str | Path | None = None,
    out_format: str = "mat",
) -> Path:
    """Resolve the shell ODB export target path."""
    from jaxmech.io.abaqus.extract_shell_odb import _resolve_out_path

    return Path(
        _resolve_out_path(
            str(odb_path),
            out_path=str(out_path) if out_path is not None else None,
            out_format=out_format,
        )
    )


def extract_shell_odb(
    odb_path: str | Path,
    *,
    out_path: str | Path | None = None,
    instance_name: str | None = None,
    out_format: str = "mat",
    abaqus_cmd: str = "abaqus",
) -> Path:
    """Extract one shell ABAQUS ODB into MAT/NPZ form via ABAQUS Python."""
    odb_path = Path(odb_path).resolve()
    script_path = PROJECT_ROOT / "jaxmech" / "io" / "abaqus" / "extract_shell_odb.py"
    _verify_wsl_accessible(odb_path, label="shell ODB file")
    _verify_wsl_accessible(script_path, label="shell ODB extractor script")

    resolved = resolve_shell_export_path(
        odb_path,
        out_path=out_path,
        out_format=out_format,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)

    win_dir = _to_windows_path_from_wsl(odb_path.parent)
    win_odb = _to_windows_path_from_wsl(odb_path)
    win_script = _to_windows_path_from_wsl(script_path)
    win_out = _to_windows_path_from_wsl(resolved)

    extra_flags = f" --out {win_out}"
    if instance_name:
        extra_flags += f" --instance {instance_name}"

    runner_path = odb_path.parent / "_jaxmech_extract_shell_odb.cmd"
    result = _run_windows_batch_script(
        runner_path,
        [
            "@echo off",
            f'cd /d {_quote_win_cmd_arg(win_dir)} || exit /b 1',
            (
                f'call {_format_win_command_name(abaqus_cmd)} '
                f'python {_quote_win_cmd_arg(win_script)} '
                f'--odb {_quote_win_cmd_arg(win_odb)} '
                f'--format {out_format}'
                f'{extra_flags}'
            ),
            "exit /b %ERRORLEVEL%",
        ],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ABAQUS shell ODB extraction failed for {odb_path.name} "
            f"(exit code {result.returncode})."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected extracted shell MAT not found: {resolved}")
    return resolved


def print_shell_extraction_commands(inp_path: str | Path) -> None:
    """Print Windows-side commands for shell ODB extraction."""
    inp = Path(inp_path).resolve()
    inp_dir = inp.parent
    inp_stem = inp.stem
    exporter = PROJECT_ROOT / "jaxmech" / "io" / "abaqus" / "extract_shell_odb.py"

    win_dir = _to_windows_path_from_wsl(inp_dir)
    win_odb = _to_windows_path_from_wsl(inp_dir / f"{inp_stem}.odb")
    win_exporter = _to_windows_path_from_wsl(exporter)

    print("\nRun these commands in Windows PowerShell:\n")
    print("  # Step 1: Run ABAQUS job if ODB does not exist yet")
    print(f"  cd {win_dir}")
    print(f"  abaqus job={inp_stem} input={inp_stem}.inp interactive")
    print()
    print("  # Step 2: Extract ODB to .mat")
    print(f"  abaqus python {win_exporter} --odb {win_odb} --format mat")


def default_shell_abaqus_mat_candidates(inp_path: str | Path) -> list[Path]:
    """Return the default MAT search locations for shell validation."""
    inp = Path(inp_path).resolve()
    model_dir = inp.parent.parent if inp.parent.name.lower() == "abaqus" else inp.parent
    return [
        model_dir / "jax" / f"{inp.stem}_shell.mat",
        model_dir / f"{inp.stem}_shell.mat",
        inp.with_suffix(".mat"),
        inp.with_name(inp.stem + "_shell.mat"),
    ]


__all__ = [
    "default_shell_abaqus_mat_candidates",
    "extract_shell_odb",
    "extract_solid_odb",
    "print_shell_extraction_commands",
    "print_solid_extraction_commands",
    "resolve_shell_export_path",
]
