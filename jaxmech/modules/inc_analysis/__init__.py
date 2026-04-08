"""
jaxmech.modules.inc_analysis — General incremental FEM analysis.

This is the standard time-domain incremental finite element analysis module.
It serves as the unified entry point for all structural analyses that can be
formulated as a sequence of load increments:

  - **Linear elastic**: Single increment, direct linear solve.
  - **Nonlinear static** (future): Newton-Raphson with multiple increments.
  - **Direct cyclic** (TODO-2): Fourier-based steady-state cyclic response.
  - **Geometrically nonlinear** (future): Updated Lagrangian / Total Lagrangian.

Architecture::

    Model  →  run_analysis(model, config)  →  AnalysisResult
                    │
                    ├── Linear elastic:   single increment, K·u = f
                    ├── Nonlinear:        Newton-Raphson loop
                    └── Direct cyclic:    Fourier iteration

Layering:
    - Incremental driver: `run_analysis()` and internal loop helpers
    - Elastic step solver: `jaxmech.modules.inc_analysis.elastic.linear.solve_linear_step`
    - Plastic driver scaffold: `jaxmech.modules.inc_analysis.plastic`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

import numpy as np

from jaxmech.materials.definition import LinearMaterialProperties, resolve_linear_material_properties
from jaxmech.model.model import Model
from jaxmech.model.result import AnalysisResult
from jaxmech.modules.inc_analysis.elastic.export_mat import (
    export_shell_elastic_result_mat,
    export_solid_elastic_result_mat,
)


@dataclass
class AnalysisConfig:
    """Configuration for an incremental analysis.

    Attributes
    ----------
    analysis_type : str
        One of: ``linear_static``, ``linear_elastic`` (alias),
        ``nonlinear_static``, ``direct_cyclic``.
    n_increments : int
        Number of load increments (1 for linear elastic).
    max_iterations : int
        Max Newton-Raphson iterations per increment (nonlinear only).
    convergence_tol : float
        Residual norm tolerance for convergence.
    gauss_order : int
        Gauss integration order (2=full, 1=reduced).
    use_b_ext : int
        B-bar formulation flag (0=standard, 1=B-bar for C3D8).
    increment_scales : list[float] or None
        Optional explicit load scales for each converged increment.
    output_vtk : str or None
        VTK output path.
    E_override, nu_override : float or None
        Material constant overrides.
    """
    analysis_type: str = "linear_static"
    n_increments: int = 1
    max_iterations: int = 20
    convergence_tol: float = 1e-8
    gauss_order: int = 2
    use_b_ext: int = 1
    solver_options: Optional[Dict[str, Any]] = None
    E_override: Optional[float] = None
    nu_override: Optional[float] = None
    yield_stress_override: Optional[float] = None
    material_model: str = "linear_elastic"
    increment_scales: Optional[list[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_analysis(
    model: Model,
    config: Optional[AnalysisConfig] = None,
    **kwargs,
) -> AnalysisResult:
    """Run a general incremental FEM analysis.

    Parameters
    ----------
    model : Model
        Unified model from ``parse_inp()``.
    config : AnalysisConfig, optional
        Analysis configuration. If None, defaults to linear elastic.
    **kwargs
        Shorthand overrides passed directly to AnalysisConfig fields.

    Returns
    -------
    AnalysisResult

    Examples
    --------
    >>> from jaxmech.io.abaqus.inp import parse_inp
    >>> from jaxmech.modules.inc_analysis import run_analysis
    >>> model = parse_inp("path/to/model.inp")
    >>> result = run_analysis(model)  # linear elastic by default
    >>> print(result.u.shape)
    """
    if config is None:
        config = AnalysisConfig(**kwargs)
    else:
        # Apply any kwargs as overrides
        for key, val in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, val)

    if config.analysis_type in ("linear_static", "linear_elastic"):
        return _run_linear_static(model, config)
    elif config.analysis_type == "nonlinear_static":
        return _run_nonlinear_static(model, config)
    elif config.analysis_type == "direct_cyclic":
        raise NotImplementedError(
            "Direct cyclic analysis is not yet implemented. "
            "See TODO-2 in the development roadmap."
        )
    else:
        raise ValueError(f"Unknown analysis_type: {config.analysis_type!r}")


def _run_incremental_loop(
    model: Model,
    config: AnalysisConfig,
    *,
    step_solver: Callable[..., AnalysisResult],
    material: LinearMaterialProperties,
) -> AnalysisResult:
    """Generic incremental driver that delegates each increment to a step solver."""
    if config.n_increments < 1:
        raise ValueError("n_increments must be >= 1.")

    increment_history = []
    result: Optional[AnalysisResult] = None

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
        increment_history.append(
            {
                "increment": increment_index,
                "load_scale": load_scale,
                "u_max": u_max,
            }
        )

    assert result is not None
    result.metadata["increment_history"] = increment_history
    result.metadata["module"] = "inc_analysis"
    return result


def _run_linear_static(model: Model, config: AnalysisConfig) -> AnalysisResult:
    """Run linear static analysis via incremental driver + single-step solver."""
    from jaxmech.modules.inc_analysis.elastic.linear import solve_linear_step

    material = resolve_linear_material_properties(
        model,
        E_override=config.E_override,
        nu_override=config.nu_override,
    )

    result = _run_incremental_loop(
        model,
        config,
        step_solver=solve_linear_step,
        material=material,
    )
    result.metadata["analysis_type"] = "linear_static"
    result.metadata["material_model"] = material.model_name
    result.metadata["source_material"] = material.source_material
    result.metadata["E"] = material.E
    result.metadata["nu"] = material.nu
    result.metadata["thickness"] = material.thickness
    return result


def _run_nonlinear_static(model: Model, config: AnalysisConfig) -> AnalysisResult:
    """Reserved entry point for future incremental nonlinear analysis."""
    if config.n_increments < 1:
        raise ValueError("n_increments must be >= 1 for nonlinear analysis.")
    try:
        from jaxmech.modules.inc_analysis.plastic import run_quasi_static_plastic
    except ImportError:
        raise ImportError(
            "弹塑性增量分析需要 jaxmech.modules.inc_analysis.plastic 模块，"
            "但该模块在当前安装中不可用。"
        )

    result = run_quasi_static_plastic(model, config)
    if result.final_snapshot is None:
        raise RuntimeError("Plastic analysis returned no final snapshot.")
    return result.final_snapshot
