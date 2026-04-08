"""
jaxmech.fem.element.shape_functions — Shape functions for reference elements.

Provides values 'N' and local derivatives 'dN_dxi' for standard reference
elements evaluated at given local coordinates.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def quad4_shape_functions(xi_eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shape functions for 4-node bilinear quadrilateral.
    Nodes ordered: (-1,-1), (1,-1), (1,1), (-1,1).

    Parameters
    ----------
    xi_eta : jnp.ndarray, shape (2,)
        Local coordinates (xi, eta) in [-1, 1].

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        N : shape (4,)
            Shape function values.
        dN_dxi : shape (2, 4)
            Derivatives with respect to local coords [d/dxi, d/deta].
    """
    xi, eta = xi_eta[0], xi_eta[1]
    
    # N_i = 1/4 * (1 +- xi) * (1 +- eta)
    N = 0.25 * jnp.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ], dtype=jnp.float64)
    
    dN_dxi = 0.25 * jnp.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)],
    ], dtype=jnp.float64)
    
    return N, dN_dxi


def tri3_shape_functions(xi_eta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shape functions for 3-node constant strain triangle.
    Nodes ordered: (0,0), (1,0), (0,1).

    Parameters
    ----------
    xi_eta : jnp.ndarray, shape (2,)
        Local coordinates (L1, L2). L3 = 1 - L1 - L2.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        N : shape (3,)
        dN_dxi : shape (2, 3)
    """
    xi, eta = xi_eta[0], xi_eta[1]
    
    N = jnp.array([
        1.0 - xi - eta,
        xi,
        eta
    ], dtype=jnp.float64)
    
    dN_dxi = jnp.array([
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ], dtype=jnp.float64)
    
    return N, dN_dxi


def hex8_shape_functions(xi_eta_zeta: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shape functions for 8-node trilinear hexahedron.
    Domain is [-1, 1]^3.

    Nodes ordered according to standard ABAQUS/Basix C3D8:
    z=-1: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1)
    z= 1: (-1,-1, 1), (1,-1, 1), (1,1, 1), (-1,1, 1)

    Parameters
    ----------
    xi_eta_zeta : jnp.ndarray, shape (3,)
        Local coordinates (xi, eta, zeta).

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        N : shape (8,)
        dN_dxi : shape (3, 8)
    """
    xi, eta, zeta = xi_eta_zeta[0], xi_eta_zeta[1], xi_eta_zeta[2]
    
    # N_i = 1/8 * (1 +- xi) * (1 +- eta) * (1 +- zeta)
    xm, xp = 1.0 - xi, 1.0 + xi
    ym, yp = 1.0 - eta, 1.0 + eta
    zm, zp = 1.0 - zeta, 1.0 + zeta
    
    N = 0.125 * jnp.array([
        xm * ym * zm, xp * ym * zm, xp * yp * zm, xm * yp * zm,
        xm * ym * zp, xp * ym * zp, xp * yp * zp, xm * yp * zp,
    ], dtype=jnp.float64)
    
    dN_dxi = 0.125 * jnp.array([
        # d/dxi
        [-ym * zm,  ym * zm,  yp * zm, -yp * zm,
         -ym * zp,  ym * zp,  yp * zp, -yp * zp],
        # d/deta
        [-xm * zm, -xp * zm,  xp * zm,  xm * zm,
         -xm * zp, -xp * zp,  xp * zp,  xm * zp],
        # d/dzeta
        [-xm * ym, -xp * ym, -xp * yp, -xm * yp,
          xm * ym,  xp * ym,  xp * yp,  xm * yp],
    ], dtype=jnp.float64)
    
    return N, dN_dxi
