"""Build incremental analysis MAT files from one inc_analysis cfg or workflow directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from jaxmech.io.abaqus.inp import parse_inp
from jaxmech.modules.inc_analysis import (
    AnalysisConfig,
    export_shell_elastic_result_mat,
    export_solid_elastic_result_mat,
    run_analysis,
)
from jaxmech.modules.inc_analysis.config import parse_inc_analysis_config
try:
    from jaxmech.modules.inc_analysis.plastic import (
        export_incremental_plastic_result_mat,
        run_quasi_static_plastic,
    )
    _HAS_PLASTIC = True
except ImportError:
    _HAS_PLASTIC = False
try:
    from jaxmech.modules.inc_analysis.shell_linear import solve_shell_elastic
    _HAS_SHELL = True
except ImportError:
    _HAS_SHELL = False


def _normalize_material_model(value: str) -> str:
    key = str(value or "auto").strip().lower()
    if key in {"", "auto", "default"}:
        return "auto"
    if key in {"linear", "elastic", "linear_static", "linear_elastic"}:
        return "linear_elastic"
    if key in {"plastic", "elastoplastic", "nonlinear", "nonlinear_static", "j2_perfect_plastic"}:
        return "j2_perfect_plastic"
    raise ValueError(f"Unsupported material_model: {value!r}")


def _resolve_requested_analysis(model, config: dict) -> tuple[str, str]:
    requested = _normalize_material_model(str(config.get("material_model", "auto")))
    family = str(model.metadata.get("family", "solid")).strip().lower()
    has_plastic = bool(model.materials and model.materials[0].plastic)
    yield_override = config.get("yield_stress_override")

    if requested == "auto":
        if family == "solid" and (has_plastic or yield_override is not None):
            return "nonlinear_static", "j2_perfect_plastic"
        return "linear_static", "linear_elastic"

    if requested == "j2_perfect_plastic":
        if family != "solid":
            raise NotImplementedError("弹塑性增量分析当前仅支持 solid 模型。")
        if not has_plastic and yield_override is None:
            raise ValueError("请求弹塑性分析，但 INP 未提供塑性材料且未设置 yield_stress_override。")
        return "nonlinear_static", "j2_perfect_plastic"

    return "linear_static", "linear_elastic"


def _config_metadata(config: dict, *, analysis_type: str, material_model: str) -> dict[str, object]:
    return {
        "requested_material_model": str(config.get("material_model", "auto")),
        "analysis_type": analysis_type,
        "material_model": material_model,
        "configured_use_b_ext": int(config.get("use_b_ext", 0)),
        "configured_gauss_order": int(config.get("gauss_order", 2)),
        "configured_n_increments": int(config.get("n_increments", 1)),
        "configured_max_iterations": int(config.get("max_iterations", 25)),
        "configured_convergence_tol": float(config.get("convergence_tol", 1.0e-8)),
        "E_override": config.get("E_override"),
        "nu_override": config.get("nu_override"),
        "yield_stress_override": config.get("yield_stress_override"),
    }


def run_jax_elastic(config_target: str | Path) -> list[Path]:
    """Parse one inc_analysis cfg/workflow, solve each INP, and write same-name MAT files."""
    config = parse_inc_analysis_config(config_target)
    workflow_dir = Path(config["workflow_dir"]).resolve()

    out_paths: list[Path] = []
    for inp_path in config["inp_files"]:
        inp_path = Path(inp_path).resolve()
        model = parse_inp(str(inp_path))
        out_mat = workflow_dir / f"{inp_path.stem}.mat"
        analysis_type, material_model = _resolve_requested_analysis(model, config)
        analysis_config = AnalysisConfig(
            analysis_type=analysis_type,
            material_model=material_model,
            use_b_ext=int(config.get("use_b_ext", 0)),
            gauss_order=int(config.get("gauss_order", 2)),
            n_increments=int(config.get("n_increments", 1)),
            max_iterations=int(config.get("max_iterations", 25)),
            convergence_tol=float(config.get("convergence_tol", 1.0e-8)),
            E_override=config.get("E_override"),
            nu_override=config.get("nu_override"),
            yield_stress_override=config.get("yield_stress_override"),
        )
        extra_metadata = {
            "config_path": str(config["config_path"]) if config.get("config_path") is not None else "",
            "load_case": inp_path.stem,
            "builder": "jaxmech.tools.build.build_model_mat",
            **_config_metadata(config, analysis_type=analysis_type, material_model=material_model),
        }

        print(f"Processing native JAX workflow for: {inp_path.stem}")
        print("  - Parsed model successfully.")

        family = str(model.metadata.get("family", "")).lower()
        if family == "shell":
            if material_model != "linear_elastic":
                raise NotImplementedError("Shell 增量分析当前仅支持弹性分支。")
            result = solve_shell_elastic(
                model,
                E_override=config.get("E_override"),
                nu_override=config.get("nu_override"),
            )
            export_shell_elastic_result_mat(
                model,
                result,
                out_mat,
                extra_metadata=extra_metadata,
            )
        elif material_model == "j2_perfect_plastic":
            if not _HAS_PLASTIC:
                raise ImportError(
                    "弹塑性分析需要 jaxmech.modules.inc_analysis.plastic 模块，"
                    "但该模块在当前安装中不可用。"
                )
            plastic_result = run_quasi_static_plastic(model, analysis_config)
            export_incremental_plastic_result_mat(
                model,
                plastic_result,
                out_mat,
                extra_metadata=extra_metadata,
            )
        else:
            result = run_analysis(model, analysis_config)
            export_solid_elastic_result_mat(
                model,
                result,
                out_mat,
                extra_metadata=extra_metadata,
            )
        print(f"  - Analyzed model successfully ({analysis_type} / {material_model}).")
        print(f"  - Exported result MAT to: {out_mat}")
        out_paths.append(out_mat)

    return out_paths


def main() -> None:
    # Force line-buffered stdout so logs stream in real time when piped.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to one inc_analysis cfg or workflow directory")
    args = parser.parse_args()

    run_jax_elastic(args.config)


if __name__ == "__main__":
    main()
