"""Build demo solid elastic MAT files from one inc_analysis cfg or directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from jaxmech.io.abaqus.inp import parse_inp
from jaxmech.modules.inc_analysis import AnalysisConfig, export_solid_elastic_result_mat, run_analysis
from jaxmech.modules.inc_analysis.config import parse_inc_analysis_config


def run_jax_elastic(config_target: str | Path) -> list[Path]:
    """Parse one cfg/workflow, solve each solid INP, and write same-name MAT files."""
    config = parse_inc_analysis_config(config_target)
    workflow_dir = Path(config["workflow_dir"]).resolve()

    out_paths: list[Path] = []
    for raw_inp_path in config["inp_files"]:
        inp_path = Path(raw_inp_path).resolve()
        model = parse_inp(str(inp_path))
        if str(model.metadata.get("family", "solid")).lower() != "solid":
            raise ValueError(f"Only solid INP files are supported in the demo: {inp_path}")

        analysis_config = AnalysisConfig(
            analysis_type="linear_static",
            use_b_ext=int(config.get("use_b_ext", 0)),
            gauss_order=int(config.get("gauss_order", 2)),
            n_increments=int(config.get("n_increments", 1)),
            E_override=config.get("E_override"),
            nu_override=config.get("nu_override"),
        )
        out_mat = workflow_dir / f"{inp_path.stem}.mat"
        extra_metadata = {
            "config_path": str(config["config_path"]) if config.get("config_path") is not None else "",
            "load_case": inp_path.stem,
            "builder": "jaxmech.tools.build.build_model_mat",
            "material_model": "linear_elastic",
            "requested_material_model": "linear_elastic",
            "configured_use_b_ext": int(config.get("use_b_ext", 0)),
            "configured_gauss_order": int(config.get("gauss_order", 2)),
            "configured_n_increments": int(config.get("n_increments", 1)),
        }

        print(f"Processing demo solid elastic workflow for: {inp_path.stem}")
        result = run_analysis(model, analysis_config)
        export_solid_elastic_result_mat(model, result, out_mat, extra_metadata=extra_metadata)
        print(f"  - Exported result MAT to: {out_mat}")
        print(f"__ARTIFACT__|mat|{out_mat}|Elastic result MAT")
        out_paths.append(out_mat)

    return out_paths


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to one inc_analysis cfg or workflow directory")
    args = parser.parse_args()
    run_jax_elastic(args.config)


if __name__ == "__main__":
    main()
