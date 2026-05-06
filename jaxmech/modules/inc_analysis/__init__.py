"""Solid linear elastic incremental analysis for the demo subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from jaxmech.materials.definition import LinearMaterialProperties, resolve_linear_material_properties
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult
from jaxmech.modules.inc_analysis.elastic.export_mat import export_solid_elastic_result_mat


@dataclass
class AnalysisConfig:
    """Configuration for the demo solid elastic analysis."""

    analysis_type: str = "linear_static"
    n_increments: int = 1
    gauss_order: int = 2
    use_b_ext: int = 1
    solver_options: Optional[dict[str, Any]] = None
    E_override: Optional[float] = None
    nu_override: Optional[float] = None
    increment_scales: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


def run_analysis(model: Model, config: Optional[AnalysisConfig] = None, **kwargs) -> AnalysisResult:
    """Run the demo-supported solid linear elastic analysis."""
    if config is None:
        config = AnalysisConfig(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if config.analysis_type not in {"linear_static", "linear_elastic"}:
        raise ValueError("The demo inc_analysis module only supports solid linear elastic analysis.")
    if str(model.metadata.get("family", "solid")).lower() != "solid":
        raise ValueError("The demo inc_analysis module only supports solid models.")
    return _run_linear_static(model, config)


def _run_incremental_loop(
    model: Model,
    config: AnalysisConfig,
    *,
    step_solver,
    material: LinearMaterialProperties,
) -> AnalysisResult:
    if config.n_increments < 1:
        raise ValueError("n_increments must be >= 1.")
    result: Optional[AnalysisResult] = None
    increment_history = []
    for increment_index in range(1, config.n_increments + 1):
        load_scale = increment_index / config.n_increments
        result = step_solver(
            model,
            material=material,
            solver_options=config.solver_options,
            gauss_order=config.gauss_order,
            use_b_ext=config.use_b_ext,
            load_scale=load_scale,
            increment_index=increment_index,
            n_increments=config.n_increments,
        )
        u_max = float(np.max(np.abs(result.u))) if result.u is not None else 0.0
        increment_history.append({"increment": increment_index, "load_scale": load_scale, "u_max": u_max})
    assert result is not None
    result.metadata["increment_history"] = increment_history
    result.metadata["module"] = "inc_analysis"
    return result


def _run_linear_static(model: Model, config: AnalysisConfig) -> AnalysisResult:
    from jaxmech.modules.inc_analysis.elastic.linear import solve_linear_step

    material = resolve_linear_material_properties(model, E_override=config.E_override, nu_override=config.nu_override)
    result = _run_incremental_loop(model, config, step_solver=solve_linear_step, material=material)
    result.metadata.update(
        {
            "analysis_type": "linear_static",
            "material_model": material.model_name,
            "source_material": material.source_material,
            "E": material.E,
            "nu": material.nu,
            "thickness": material.thickness,
        }
    )
    return result


__all__ = ["AnalysisConfig", "run_analysis", "export_solid_elastic_result_mat"]
