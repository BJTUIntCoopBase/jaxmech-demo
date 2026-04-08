"""
jaxmech.fem.element.core — Core abstractions for finite element kernels.

Every element formulation (e.g. C3D8, CPS4, STRI3) must subclass ``ElementKernel``
and implement its required methods. These methods must be compatible with JAX
transformations (e.g. ``jax.vmap``, ``jax.jit``).

Design principle: ElementKernel provides **pure differential geometry / kinematics
operators only**. No material constitutive logic appears here. Material-dependent
computations (K = B^T D B, σ = D ε) live in the assembly layer.

Voigt conventions
-----------------
- ``voigt_E`` (engineering): shear components use γ = 2ε.
  Paired with the standard elastic D matrix (D^E).
- ``voigt_S`` (tensor):      shear components use ε (no factor of 2).
  Paired with D^S = P⁻¹ D^E P⁻¹, where P = diag(1,...,1,½,...,½).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import jax.numpy as jnp


class ElementKernel(ABC):
    """Base class for finite element kernels.

    Subclasses implement specific element formulations and must provide
    pure geometric / kinematics methods.  Material constitutive logic
    is handled by the assembly layer, not here.

    Attributes
    ----------
    ele_type : str
        Element type name (e.g. 'C3D8', 'CPS4').
    cell_type : str
        Cell geometry type (e.g. 'hexahedron', 'quadrangle').
    n_nodes : int
        Number of nodes per element.
    n_dim : int
        Spatial dimension (2 or 3).  For shells with 6 DOF/node, this is 6.
    n_gauss : int
        Number of Gauss integration points.
    n_str : int
        Number of stress/strain components (e.g. 3 for 2D, 6 for 3D).
    """

    ele_type: str = ""
    cell_type: str = ""
    n_nodes: int = 0
    n_dim: int = 0
    n_gauss: int = 0
    n_str: int = 0

    # ------------------------------------------------------------------
    # Abstract methods — every kernel MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def jacobian(
        self, node_coords: jnp.ndarray, gp_idx: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute the Jacobian mapping at a Gauss point.

        Parameters
        ----------
        node_coords : (n_nodes, n_spatial_dim)
        gp_idx : int

        Returns
        -------
        J      : Jacobian matrix
        J_inv  : Inverse Jacobian
        detJ   : Determinant of J (scalar)
        """

    @abstractmethod
    def sym_grad_voigt_E(
        self, node_coords: jnp.ndarray, gp_idx: int
    ) -> jnp.ndarray:
        """Symmetric gradient operator in *engineering* Voigt form.

        Shear rows contain γ_ij = 2 ε_ij.

        Returns
        -------
        B_E : (n_str, n_dof_elem)
        """

    @abstractmethod
    def gauss_info(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Gauss quadrature points and weights.

        Returns
        -------
        points  : (n_gauss, n_parent_dim)
        weights : (n_gauss,)
        """

    # ------------------------------------------------------------------
    # Default implementations
    # ------------------------------------------------------------------

    def sym_grad_voigt_S(
        self, node_coords: jnp.ndarray, gp_idx: int
    ) -> jnp.ndarray:
        """Symmetric gradient operator in *tensor* Voigt form (no 2×).

        Default: P @ B_E  where P = diag(1,...,1, ½,...,½).

        Returns
        -------
        B_S : (n_str, n_dof_elem)
        """
        B_E = self.sym_grad_voigt_E(node_coords, gp_idx)
        # Build P: first components (normal strains) are 1, shear are 0.5
        if self.n_str == 3:
            # 2D: [ε11, ε22, γ12] → [ε11, ε22, ε12]
            p = jnp.array([1.0, 1.0, 0.5], dtype=jnp.float64)
        elif self.n_str == 6:
            # 3D: [ε11, ε22, ε33, γ12, γ23, γ13] → tensor form
            p = jnp.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=jnp.float64)
        else:
            p = jnp.ones(self.n_str, dtype=jnp.float64)
        return p[:, None] * B_E

    def b_matrix(
        self, node_coords: jnp.ndarray, gp_idx: int
    ) -> jnp.ndarray:
        """Backward-compatible alias for ``sym_grad_voigt_E``."""
        return self.sym_grad_voigt_E(node_coords, gp_idx)

    def sym_grad_voigt_E_all(self, node_coords: jnp.ndarray) -> jnp.ndarray:
        """Return engineering-Voigt B matrices for all Gauss points.

        Subclasses can override this to provide a more efficient batched
        implementation. The default path vectorizes ``sym_grad_voigt_E`` over
        all Gauss points.
        """
        gp_ids = jnp.arange(self.n_gauss, dtype=jnp.int32)
        return jax.vmap(lambda gp_idx: self.sym_grad_voigt_E(node_coords, gp_idx))(gp_ids)
