"""jaxmech.fem.solver — Linear system solvers for FEM (material-independent).

Public API:
  - linear_solve(K, f, solver_options) — dispatches to scipy/jax/petsc/custom
"""

from jaxmech.fem.solver.linear import linear_solve
