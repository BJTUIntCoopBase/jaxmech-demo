"""
jaxmech - A JAX-based general-purpose computational mechanics library.

Modules:
    io        - External format I/O (ABAQUS inp/odb, meshio, etc.)
    model     - Unified data model (Mesh, Model, AnalysisResult, ...)
    fem       - Finite element core (elements, quadrature, assembly, solver)
    materials - Constitutive models (elastic, plasticity, crystal plasticity, ...)
    modules   - High-level analysis modules (shakedown, validation, topopt, ...)
    optimize  - Optimization backend interfaces (CVXPY, Gurobi, ...)
"""

from jaxmech.env.env import ensure_env_cfg_exists

__version__ = "0.1.0-dev"

ensure_env_cfg_exists()
