"""
jaxmech.fem.solver.linear — Linear system solver dispatch.

Provides a unified ``linear_solve(K, f, solver_options)`` interface that
dispatches to different backends:

- ``scipy``  — scipy.sparse.linalg.spsolve (default, direct)
- ``jax``    — jax.scipy.sparse.linalg.bicgstab (iterative, GPU-ready)
- ``petsc``  — PETSc KSP solver (if petsc4py available)
- ``custom`` — user-provided callable(K, f) -> u

Inspired by jax-fem's ``linear_solver(A, b, x0, solver_options)`` pattern.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
#  Backend availability checks
# ---------------------------------------------------------------------------

try:
    from petsc4py import PETSc
    _PETSC_AVAILABLE = True
except ImportError:
    _PETSC_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Individual solver backends
# ---------------------------------------------------------------------------

def _scipy_solve(K: sp.csr_matrix, f: np.ndarray) -> np.ndarray:
    """Direct solve via SciPy UMFPACK / SuperLU."""
    return spsolve(K, f)


def _jax_solve(
    K: sp.csr_matrix,
    f: np.ndarray,
    *,
    precond: bool = True,
    tol: float = 1e-10,
    maxiter: int = 10000,
) -> np.ndarray:
    """Iterative solve via JAX's BiCGSTAB (CPU or GPU).

    Parameters
    ----------
    precond : bool
        If True, use Jacobi (diagonal) preconditioning.
    tol, maxiter : float, int
        Convergence parameters for BiCGSTAB.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("JAX is not available. Install jax to use jax solver.")

    from jax.experimental.sparse import BCOO

    # Convert scipy CSR to JAX BCOO
    K_coo = K.tocoo()
    indices = jnp.stack([jnp.array(K_coo.row), jnp.array(K_coo.col)], axis=-1)
    data = jnp.array(K_coo.data, dtype=jnp.float64)
    K_jax = BCOO((data, indices), shape=K.shape)

    f_jax = jnp.array(f, dtype=jnp.float64)

    # Preconditioner (Jacobi / diagonal)
    M = None
    if precond:
        diag = jnp.array(K.diagonal(), dtype=jnp.float64)
        diag = jnp.where(jnp.abs(diag) < 1e-15, 1.0, diag)

        def M_fn(x):
            return x / diag

        M = M_fn

    # Solve with BiCGSTAB
    @jax.jit
    def _solve(K_jax, f_jax):
        def matvec(x):
            return K_jax @ x
        u, info = jax.scipy.sparse.linalg.bicgstab(
            matvec, f_jax, tol=tol, maxiter=maxiter, M=M
        )
        return u

    u = _solve(K_jax, f_jax)
    return np.asarray(u)


def _petsc_solve(
    K: sp.csr_matrix,
    f: np.ndarray,
    *,
    ksp_type: str = "bcgsl",
    pc_type: str = "ilu",
) -> np.ndarray:
    """Solve via PETSc KSP.

    Parameters
    ----------
    ksp_type : str
        KSP solver type (e.g. 'cg', 'bcgsl', 'gmres', 'minres').
    pc_type : str
        Preconditioner type (e.g. 'ilu', 'jacobi', 'none').
    """
    if not _PETSC_AVAILABLE:
        raise RuntimeError(
            "PETSc is not available. Install petsc4py to use PETSc solver."
        )

    # Create PETSc matrix from CSR
    K_csr = K.tocsr()
    A = PETSc.Mat().createAIJ(
        size=K_csr.shape,
        csr=(
            K_csr.indptr.astype(PETSc.IntType, copy=False),
            K_csr.indices.astype(PETSc.IntType, copy=False),
            K_csr.data,
        ),
    )

    # Create vectors
    b = PETSc.Vec().createWithArray(f)
    x = PETSc.Vec().createWithArray(np.zeros_like(f))

    # Setup KSP
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)
    ksp.setFromOptions()
    ksp.solve(b, x)

    result = x.getArray().copy()

    # Cleanup
    ksp.destroy()
    A.destroy()
    b.destroy()
    x.destroy()

    return result


# ---------------------------------------------------------------------------
#  Public dispatch function
# ---------------------------------------------------------------------------

def linear_solve(
    K: sp.csr_matrix,
    f: np.ndarray,
    solver_options: Optional[dict] = None,
) -> np.ndarray:
    """Solve the linear system K u = f.

    Parameters
    ----------
    K : scipy.sparse.csr_matrix
        Stiffness matrix (already reduced to free DOFs).
    f : np.ndarray
        Right-hand side (load vector for free DOFs).
    solver_options : dict, optional
        Solver selection and configuration.  The **key** determines the
        backend; the **value** (a dict) is passed as kwargs to it.

        Supported keys::

            {'scipy': {}}                                   # default
            {'jax': {'precond': True, 'tol': 1e-10}}
            {'petsc': {'ksp_type': 'cg', 'pc_type': 'ilu'}}
            {'custom': my_solver_callable}

        If ``None`` or ``{}``, defaults to ``{'scipy': {}}``.

        For ``'custom'``, the value must be a callable with
        signature ``(K: csr_matrix, f: ndarray) -> ndarray``.

    Returns
    -------
    u : np.ndarray
        Solution vector.
    """
    if solver_options is None or len(solver_options) == 0:
        solver_options = {"scipy": {}}

    if "scipy" in solver_options:
        return _scipy_solve(K, f)

    if "jax" in solver_options:
        opts = solver_options["jax"] if isinstance(solver_options["jax"], dict) else {}
        return _jax_solve(K, f, **opts)

    if "petsc" in solver_options:
        opts = solver_options["petsc"] if isinstance(solver_options["petsc"], dict) else {}
        return _petsc_solve(K, f, **opts)

    if "custom" in solver_options:
        custom_fn = solver_options["custom"]
        if not callable(custom_fn):
            raise TypeError(
                f"'custom' solver_options value must be callable, got {type(custom_fn)}"
            )
        return custom_fn(K, f)

    raise ValueError(
        f"Unknown solver backend(s): {list(solver_options.keys())}. "
        f"Supported: 'scipy', 'jax', 'petsc', 'custom'."
    )
