"""Isotropic linear elastic constitutive models and D-matrix helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np

from jaxmech.materials.base import ConstitutiveContext, ConstitutiveModel


def d_matrix_plane_stress(E: float, nu: float) -> jnp.ndarray:
    """Isotropic plane stress D matrix with shape ``(3, 3)``."""
    c = E / (1.0 - nu**2)
    return c * jnp.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=jnp.float64,
    )


def d_matrix_isotropic_3d(E: float, nu: float) -> jnp.ndarray:
    """Isotropic 3D D matrix with shape ``(6, 6)``."""
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    c1 = 1.0 - nu
    c2 = (1.0 - 2.0 * nu) / 2.0
    return c * jnp.array(
        [
            [c1, nu, nu, 0.0, 0.0, 0.0],
            [nu, c1, nu, 0.0, 0.0, 0.0],
            [nu, nu, c1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c2],
        ],
        dtype=jnp.float64,
    )


def d_matrix_shell_generalized(E: float, nu: float, thickness: float) -> jnp.ndarray:
    """Shell generalized D matrix with shape ``(6, 6)``."""
    cm = E / (1.0 - nu**2)
    membrane = cm * jnp.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=jnp.float64,
    ) * thickness

    cb = E * thickness**3 / (12.0 * (1.0 - nu**2))
    bending = cb * jnp.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=jnp.float64,
    )

    D = jnp.zeros((6, 6), dtype=jnp.float64)
    D = D.at[0:3, 0:3].set(membrane)
    D = D.at[3:6, 3:6].set(bending)
    return D


def get_d_matrix(material_props: dict, kernel) -> jnp.ndarray:
    """Build the elastic stiffness matrix based on kernel family."""
    E = material_props["E"]
    nu = material_props["nu"]

    if kernel.ele_type == "STRI3":
        thickness = material_props.get("thickness", 1.0)
        return d_matrix_shell_generalized(E, nu, thickness)

    if kernel.n_str == 3:
        return d_matrix_plane_stress(E, nu)

    return d_matrix_isotropic_3d(E, nu)


class IsotropicElastic(ConstitutiveModel):
    """Small-strain isotropic linear elastic constitutive model."""

    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    @property
    def name(self) -> str:
        return "IsotropicElastic"

    @property
    def n_state_vars(self) -> int:
        return 0

    def update(
        self,
        strain: np.ndarray,
        state: Optional[np.ndarray],
        ctx: ConstitutiveContext,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        D = self.elastic_stiffness(ctx)
        stress = D @ strain
        return stress, D, None

    def elastic_stiffness(self, ctx: ConstitutiveContext) -> jnp.ndarray:
        if ctx.plane == "stress":
            return d_matrix_plane_stress(self.E, self.nu)
        return d_matrix_isotropic_3d(self.E, self.nu)