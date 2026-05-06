"""CLI entry point for elastic ODB validation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from jaxmech.env.env import PROJECT_ROOT
from jaxmech.io.abaqus.odb import extract_solid_odb
from jaxmech.modules.validation.elastic.compare import ValidationReport
from jaxmech.modules.validation.elastic.config import parse_validation_config, _infer_family_from_mat
from jaxmech.modules.validation.elastic.solid_odb import (
    compute_validation_error_distributions_from_mat,
    validate_mat_with_abaqus_odb,
)


def _copy_or_reuse_mat(src_mat: Path, workflow_dir: Path, copy_mat: bool) -> Path:
    if not copy_mat:
        return src_mat
    dst_mat = workflow_dir / src_mat.name
    if src_mat.resolve() != dst_mat.resolve():
        shutil.copy2(src_mat, dst_mat)
    return dst_mat


def _default_validation_workflow_dir(model_root: Path | None, mat_path: Path) -> Path:
    if model_root is not None:
        return model_root / "validation"
    parent = mat_path.parent
    if parent.name.lower() == "inc_analysis":
        return parent.parent / "validation"
    return parent / "validation"


def _default_temp_validation_dir(workflow_dir: Path, mat_path: Path) -> Path:
    model_name = workflow_dir.parent.name if workflow_dir.name.lower() == "validation" else workflow_dir.name
    return PROJECT_ROOT / "TempTest" / "validation_odb" / model_name / mat_path.stem


def _emit_artifact(kind: str, path: Path, label: str) -> None:
    print(f"__ARTIFACT__|{kind}|{path.resolve()}|{label}")


def _boxplot_series_for_report(
    family: str,
    validated_mat: Path,
    abaqus_mat_path: str | Path | None,
    errors: dict[str, float],
) -> dict[str, np.ndarray]:
    family_key = str(family).strip().lower()
    if family_key == "solid" and abaqus_mat_path:
        try:
            return compute_validation_error_distributions_from_mat(validated_mat, abaqus_mat_path)
        except Exception as exc:
            print(f"[Validation] boxplot 分布提取失败，回退到标量误差: {exc}")
    return {
        key: np.asarray([float(val)], dtype=np.float64)
        for key, val in errors.items()
        if isinstance(val, (int, float))
    }


def _write_validation_boxplot(
    out_path: Path,
    *,
    family: str,
    validated_mat: Path,
    abaqus_mat_path: str | Path | None,
    errors: dict[str, float],
) -> Path | None:
    series = _boxplot_series_for_report(family, validated_mat, abaqus_mat_path, errors)
    labels: list[str] = []
    data: list[np.ndarray] = []
    for key in sorted(series):
        arr = np.asarray(series[key], dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        labels.append(f"{key}\n(n={arr.size})")
        data.append(arr)
    if not data:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Validation] matplotlib 不可用，跳过 boxplot: {exc}")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(6.5, 1.6 * len(data))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    palette = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#7c3aed", "#06b6d4"]
    for idx, patch in enumerate(box["boxes"]):
        patch.set(facecolor=palette[idx % len(palette)], alpha=0.45, edgecolor="#cbd5e1")
    for median in box.get("medians", []):
        median.set(color="#f8fafc", linewidth=1.6)
    for whisker in box.get("whiskers", []):
        whisker.set(color="#94a3b8", linewidth=1.0)
    for cap in box.get("caps", []):
        cap.set(color="#94a3b8", linewidth=1.0)

    positives = np.concatenate([arr[arr > 0.0] for arr in data if np.any(arr > 0.0)], axis=0) if any(np.any(arr > 0.0) for arr in data) else np.asarray([], dtype=np.float64)
    if positives.size and positives.max() / max(positives.min(), 1.0e-16) > 1.0e3:
        ax.set_yscale("log")

    ax.set_title(f"{validated_mat.stem} ODB Validation Boxplot", fontsize=12)
    ax.set_ylabel("Row-wise L1 absolute error", fontsize=10)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run_elastic_validation_from_config(config: dict, *, force: bool = False) -> list[ValidationReport]:
    workflow_dir = Path(config["workflow_dir"]).resolve()
    workflow_dir.mkdir(parents=True, exist_ok=True)
    family = str(config["family"]).lower()
    copy_mat = bool(config["copy_mat"])
    abaqus_cmd = str(config.get("windows_abaqus_cmd", "")).strip()
    if not abaqus_cmd:
        raise RuntimeError("validation requires windows_abaqus_cmd in the validation configuration.")

    mat_files = [Path(p).resolve() for p in config["mat_files"]]
    odb_files = [Path(p).resolve() for p in config["odb_files"]]
    if len(mat_files) != len(odb_files):
        raise RuntimeError("validation mat_files / odb_files length mismatch.")

    print(f"[Validation] 阶段 1/5 开始：准备工作目录 {workflow_dir}")
    print(f"[Validation] family={family}, cases={len(mat_files)}, copy_mat={int(copy_mat)}, force={int(force)}")
    if family != "solid":
        raise RuntimeError("jaxmech-demo only includes solid elastic ODB validation.")
    print("[Validation] 阶段 1/5 完成：输入检查通过。")

    reports: list[ValidationReport] = []
    for case_index, (src_mat, odb_path) in enumerate(zip(mat_files, odb_files), start=1):
        if not src_mat.is_file():
            raise FileNotFoundError(f"Validation source MAT not found: {src_mat}")
        if not odb_path.is_file():
            raise FileNotFoundError(f"Validation target ODB not found: {odb_path}")

        print(f"[Validation] ===== 算例 {case_index}/{len(mat_files)}: {src_mat.name} =====")
        print(f"[Validation] 阶段 2/5 开始：准备 validation MAT")
        work_mat = _copy_or_reuse_mat(src_mat, workflow_dir, copy_mat)
        if work_mat.resolve() == src_mat.resolve():
            print(f"[Validation] 阶段 2/5 完成：复用源 MAT {work_mat}")
        else:
            print(f"[Validation] 阶段 2/5 完成：结果 MAT 已放入 {work_mat}")

        raw: dict
        abaqus_mat_path: str | Path | None = None
        temp_out_dir: Path | None = None

        try:
            temp_out_dir = _default_temp_validation_dir(workflow_dir, src_mat)
            temp_out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Validation] 阶段 3/5 开始：提取 solid ODB 到临时目录 {temp_out_dir}")
            abaqus_mat_path = extract_solid_odb(
                win_job_dir=str(odb_path.parent),
                odb_name=odb_path.name,
                abaqus_cmd=abaqus_cmd,
                linux_out_dir=str(temp_out_dir),
                skip_initial=False,
                step=None,
                fixed_dof1_set=None,
                fixed_dof2_set=None,
                fixed_dof3_set=None,
            )
            print(f"[Validation] 阶段 3/5 完成：ABAQUS MAT 已生成 {abaqus_mat_path}")
            print("[Validation] 阶段 4/5 开始：执行字段比对并写回 validation MAT")
            raw = validate_mat_with_abaqus_odb(
                work_mat,
                force=force,
                abaqus_mat_path=abaqus_mat_path,
                validated_odb_name=str(odb_path),
            )
            print(f"[Validation] 阶段 4/5 完成：validation MAT 已写入 {work_mat}")

            _emit_artifact("mat", work_mat, f"Validation MAT: {work_mat.stem}")

            errors = raw.get("errors", {})
            err_dict = {
                key: float(val)
                for key, val in errors.items()
                if isinstance(val, (int, float))
            } if isinstance(errors, dict) else {}

            print("[Validation] 当前误差摘要：")
            for key, val in sorted(err_dict.items()):
                print(f"[Validation]   max_norm1({key}) = {val:.6e}")

            plot_path = workflow_dir / f"{work_mat.stem}_validation_boxplot.png"
            print(f"[Validation] 阶段 5/5 开始：生成 boxplot -> {plot_path}")
            plot_written = _write_validation_boxplot(
                plot_path,
                family=family,
                validated_mat=work_mat,
                abaqus_mat_path=raw.get("abaqus_mat_path") or abaqus_mat_path,
                errors=err_dict,
            )
            if plot_written is not None:
                print(f"[Validation] 阶段 5/5 完成：boxplot 已写入 {plot_written}")
                _emit_artifact("image", plot_written, f"Validation Boxplot: {work_mat.stem}")
            else:
                print("[Validation] 阶段 5/5 完成：当前算例未生成 boxplot。")

            reports.append(
                ValidationReport(
                    errors=err_dict,
                    family=family,
                    metadata={
                        "source_mat": str(src_mat),
                        "validated_mat": str(work_mat),
                        "odb_path": str(odb_path),
                        "abaqus_mat_path": str(raw.get("abaqus_mat_path") or abaqus_mat_path or ""),
                        "skipped": raw.get("skipped", False),
                        "boxplot_path": str(plot_written) if plot_written is not None else "",
                    },
                )
            )
        finally:
            if temp_out_dir is not None and temp_out_dir.exists():
                shutil.rmtree(temp_out_dir, ignore_errors=True)
                print(f"[Validation] 已清理临时目录 {temp_out_dir}")

    print("[Validation] 全部阶段完结。")
    return reports


def run_elastic_validation(
    config_target: str | Path,
    *,
    force: bool = False,
) -> list[ValidationReport]:
    """Run elastic validation from one validation cfg or workflow directory."""
    config = parse_validation_config(config_target)
    return _run_elastic_validation_from_config(config, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ── cfg-based usage (original) ──
    parser.add_argument("--config", default=None, help="Path to one validation cfg or workflow directory.")
    parser.add_argument("--force", action="store_true", help="Re-run even if already validated.")
    # ── direct usage (new — skips cfg file) ──
    parser.add_argument("--mat", default=None, help="Direct: WSL path to the elastic MAT file.")
    parser.add_argument("--odb", default=None, help="Direct: WSL path to the ABAQUS ODB file.")
    parser.add_argument("--model-root", default=None, help="Direct: WSL path to the model root (used as workflow_dir).")
    parser.add_argument("--validation-dir", default=None, help="Direct: WSL path to the validation output directory.")
    parser.add_argument("--abaqus-cmd", default=None, help="Direct: Windows ABAQUS command (overrides env.cfg).")
    parser.add_argument("--copy-mat", action="store_true", help="Copy MAT into workflow dir before validation.")
    args = parser.parse_args()

    if args.mat:
        mat_p = Path(args.mat).resolve()
        odb_p = Path(args.odb).resolve() if args.odb else None
        if odb_p is None:
            raise SystemExit("--odb is required when using --mat direct mode.")

        # Resolve abaqus_cmd: CLI > env.cfg
        abaqus_cmd = (args.abaqus_cmd or "").strip()
        if not abaqus_cmd:
            try:
                from jaxmech.env.env import load_machine_config
                env = load_machine_config()
                abaqus_cmd = str(env.get("windows_abaqus_cmd", "")).strip()
            except Exception:
                pass

        if not abaqus_cmd:
            raise SystemExit(
                "windows_abaqus_cmd not found. Pass --abaqus-cmd or set windows_abaqus_cmd in env.cfg."
            )

        model_root = Path(args.model_root).resolve() if args.model_root else None
        workflow_dir = Path(args.validation_dir).resolve() if args.validation_dir else _default_validation_workflow_dir(model_root, mat_p)
        workflow_dir.mkdir(parents=True, exist_ok=True)

        family = _infer_family_from_mat(mat_p)
        config = {
            "mat_files": [mat_p],
            "odb_files": [odb_p],
            "copy_mat": args.copy_mat,
            "windows_abaqus_cmd": abaqus_cmd,
            "workflow_dir": workflow_dir,
            "model_dir": workflow_dir,
            "family": family,
        }
        reports = _run_elastic_validation_from_config(config, force=args.force)
        for report in reports:
            print(report.summary())
    elif args.config:
        reports = run_elastic_validation(Path(args.config), force=args.force)
        for report in reports:
            print(report.summary())
    else:
        parser.error("Either --config or --mat/--odb must be provided.")


if __name__ == "__main__":
    main()

