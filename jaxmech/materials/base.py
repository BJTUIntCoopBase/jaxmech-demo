"""
jaxmech.materials.base - Abstract base class for constitutive models.

A ConstitutiveModel encapsulates the material stress-strain law.
Its unified interface is:

    update(strain, state, context) -> (stress, tangent, state_new)

This makes it compatible with:
  - Linear elastic analysis (tangent = const, state = None)
  - J2 plasticity (tangent = algorithmic, state = back stress + eps_p)
  - Crystal plasticity (state = orientation + slip system variables)
  - Viscoelasticity (state = internal variables, context has dt)

All implementations should be JAX-compatible (jit/vmap friendly).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class ConstitutiveContext:
    """Context information passed to constitutive update.

    Attributes:
        dt:        Time step size (for rate-dependent models).
        temperature: Current temperature (for thermal coupling).
        n_str:     Number of stress/strain components (3 or 6).
        plane:     Plane assumption: "stress", "strain", or None (3D).
        extra:     Any extra context data.
    """
    dt: float = 0.0
    temperature: float = 0.0
    n_str: int = 6
    plane: Optional[str] = None  # "stress", "strain", or None for 3D
    extra: Dict[str, Any] = field(default_factory=dict)


class ConstitutiveModel(ABC):
    """Abstract base class for material constitutive models.

    Subclasses implement specific constitutive laws:
      - IsotropicElastic
      - J2Plasticity
      - CrystalPlasticity
      - OrthotropicElastic
      - ShellLaminateConstitutive
      - ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def n_state_vars(self) -> int:
        """Number of internal state variables per integration point.

        Return 0 for stateless models (e.g. linear elastic).
        """
        ...

    @abstractmethod
    def update(
        self,
        strain: np.ndarray,
        state: Optional[np.ndarray],
        ctx: ConstitutiveContext,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute stress, tangent modulus, and updated internal state.

        This is the core constitutive interface. All analysis modules
        call this method to get the material response.

        Args:
            strain: Strain tensor in Voigt notation, shape (n_str,).
            state:  Internal state variables, shape (n_state_vars,),
                    or None for the initial call / stateless models.
            ctx:    Context information (time step, temperature, etc.).

        Returns:
            stress:    Stress tensor in Voigt notation, shape (n_str,).
            tangent:   Consistent tangent modulus, shape (n_str, n_str).
            state_new: Updated internal state, shape (n_state_vars,),
                       or None for stateless models.
        """
        ...

    def elastic_stiffness(self, ctx: ConstitutiveContext) -> np.ndarray:
        """Return the elastic stiffness matrix (for linear problems).

        Optional convenience method. Default calls update with zero strain.

        Args:
            ctx: Context with n_str and plane information.

        Returns:
            D: Elastic stiffness matrix, shape (n_str, n_str).
        """
        zero_strain = np.zeros(ctx.n_str)
        _, D, _ = self.update(zero_strain, None, ctx)
        return D
