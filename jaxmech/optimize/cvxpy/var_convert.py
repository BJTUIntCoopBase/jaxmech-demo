"""
var_convert.py
==============

Variable-conversion utilities shared by the shakedown solvers.

This module now exposes two clearly separated layers:

1. Version-C interface
   Uses a quadratic von Mises factor ``P`` such that
   ``||P @ sigma||^2 = sigma_eq^2``.

2. Version-MN legacy interface
   Reproduces the legacy MATLAB/Gurobi stress-variable conversion exactly:
   ``Sig2UX*``, ``UX2Sig*``, ``S2UMat`` and the equilibrium-block
   reconstruction used in ``VarConvert3D.m`` / ``VarConvertCPE.m`` /
   ``ExtractCPSConvertProcess.m``.

For Version-MN, the intent is strict compatibility with the legacy MATLAB/Gurobi
pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Legacy transform definitions
# ---------------------------------------------------------------------------

T_INV_3D = np.array([
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, np.sqrt(6.0), 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, np.sqrt(6.0), 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(6.0)],
], dtype=np.float64)

L_TRANS_3D = np.array([
    [np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0, 0.0],
    [0.0, np.sqrt(3.0) / np.sqrt(2.0), 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

T_INV_CPE = np.array([
    [1.0, -1.0, 0.0, 0.0],
    [0.0, 1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, np.sqrt(6.0)],
], dtype=np.float64)

L_TRANS_CPE = np.array([
    [np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
    [0.0, np.sqrt(3.0) / np.sqrt(2.0), 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

INV_L_TRANS_CPS = np.array([
    [1.0, np.sqrt(3.0) / 3.0, 0.0],
    [0.0, 2.0 * np.sqrt(3.0) / 3.0, 0.0],
    [0.0, 0.0, np.sqrt(3.0) / 3.0],
], dtype=np.float64)


def _as_float_vector(name: str, values, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must have length {size}, got {arr.size}.")
    return arr


class VarConverter:
    """Element-type specific stress-variable conversions.

    The public API is intentionally split:

    Version-C:
    - ``get_J2_quadratic_matrix()``
    - ``get_J2_matrix()``
    - ``equiv_stress()``
    - ``yield_ratio()``

    Version-MN / legacy:
    - ``get_legacy_s2u_matrix()``
    - ``get_legacy_sig2ux_matrix(sigma_y)``
    - ``get_legacy_ux2sig_matrix(sigma_y)``
    - ``sig2ux(sigma, sigma_y)``
    - ``ux2sig(U, X, sigma_y)``
    - ``convert_c_matrix_block_legacy(c_block, sigma_y)``
    """

    def __init__(self, ele_type: str):
        self.ele_type = ele_type
        if ele_type == "C3D8":
            self._setup_3d()
        elif ele_type == "CPE4":
            self._setup_cpe()
        elif ele_type in ("CPS4", "CPS4R", "CPS3"):
            self._setup_cps()
        else:
            raise ValueError(f"Unsupported element type: {ele_type}")

    # ------------------------------------------------------------------
    # Legacy setup
    # ------------------------------------------------------------------

    def _setup_3d(self) -> None:
        self.n_str = 6
        self.n_U = 5
        self.n_X = 1

        self._T_inv = T_INV_3D
        self._T = np.linalg.inv(T_INV_3D)
        self.B_row = T_INV_3D[2:3, :]

        dev_rows = np.vstack([T_INV_3D[0:2, :], T_INV_3D[3:6, :]])
        self.S2U = (1.0 / np.sqrt(2.0)) * L_TRANS_3D @ dev_rows

        self._legacy_sig2ux_unit = np.vstack([self.S2U, self.B_row])
        self._legacy_ux2sig_unit = np.linalg.inv(self._legacy_sig2ux_unit)

    def _setup_cpe(self) -> None:
        self.n_str = 4
        self.n_U = 3
        self.n_X = 1

        self._T_inv = T_INV_CPE
        self._T = np.linalg.inv(T_INV_CPE)
        self.B_row = T_INV_CPE[2:3, :]

        dev_rows = np.vstack([T_INV_CPE[0:2, :], T_INV_CPE[3:4, :]])
        self.S2U = (1.0 / np.sqrt(2.0)) * L_TRANS_CPE @ dev_rows

        self._legacy_sig2ux_unit = np.vstack([self.S2U, self.B_row])
        self._legacy_ux2sig_unit = np.linalg.inv(self._legacy_sig2ux_unit)

    def _setup_cps(self) -> None:
        self.n_str = 3
        self.n_U = 3
        self.n_X = 0

        self._T_inv = None
        self._T = None
        self.B_row = None

        # Matches ExtractCPSConvertProcess.m exactly:
        # InvL_MatTrans = [...]
        # S2UMat = inv(InvL_MatTrans)
        self.S2U = np.linalg.inv(INV_L_TRANS_CPS)

        self._legacy_sig2ux_unit = self.S2U.copy()
        self._legacy_ux2sig_unit = INV_L_TRANS_CPS.copy()

    # ------------------------------------------------------------------
    # Version-C interface
    # ------------------------------------------------------------------

    def get_J2_quadratic_matrix(self) -> np.ndarray:
        """Return ``Q`` such that ``sigma_eq^2 = sigma.T @ Q @ sigma``.

        This is built directly from the legacy ``S2UMat`` so that the
        Version-C quadratic form and the Version-MN cone form share the
        same underlying von Mises definition.
        """
        return self.S2U.T @ self.S2U

    def get_J2_matrix(self) -> np.ndarray:
        """Return ``P`` such that ``||P @ sigma||^2 = sigma_eq^2``.

        We return the legacy-aligned ``S2UMat`` directly. This keeps the
        Version-C QCQP factor and the future Version-MN SOCP factor
        numerically identical.
        """
        return self.S2U.copy()

    def equiv_stress(self, sigma: np.ndarray) -> float:
        """Compute von Mises equivalent stress from the legacy ``S2UMat``."""
        sigma = _as_float_vector("sigma", sigma, self.n_str)
        return float(np.linalg.norm(self.S2U @ sigma))

    def yield_ratio(self, sigma: np.ndarray, sigma_y: float) -> float:
        sigma_y = float(sigma_y)
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}.")
        return self.equiv_stress(sigma) / sigma_y

    # ------------------------------------------------------------------
    # Version-MN / legacy interface
    # ------------------------------------------------------------------

    def get_legacy_s2u_matrix(self) -> np.ndarray:
        """Return the legacy ``S2UMat`` for unit yield strength."""
        return self.S2U.copy()

    def get_legacy_sig2ux_matrix(self, sigma_y: float) -> np.ndarray:
        """Return the exact legacy ``Sig2UX*`` matrix for a Gauss point.

        For 3D/CPE the result maps stress to ``[U; X]``.
        For CPS the result maps stress to ``U`` only.
        """
        sigma_y = float(sigma_y)
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}.")
        if self.n_X == 0:
            return self._legacy_sig2ux_unit / sigma_y
        top = self.S2U / sigma_y
        bottom = self.B_row.copy()
        return np.vstack([top, bottom])

    def get_legacy_ux2sig_matrix(self, sigma_y: float) -> np.ndarray:
        """Return the exact legacy ``UX2Sig*`` matrix for a Gauss point."""
        sigma_y = float(sigma_y)
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}.")
        if self.n_X == 0:
            return sigma_y * self._legacy_ux2sig_unit
        sig2ux = self.get_legacy_sig2ux_matrix(sigma_y)
        return np.linalg.inv(sig2ux)

    def sig2ux(self, sigma: np.ndarray, sigma_y: float) -> Tuple[np.ndarray, np.ndarray | None]:
        """Legacy-compatible stress -> ``(U, X)`` conversion."""
        sigma = _as_float_vector("sigma", sigma, self.n_str)
        ux = self.get_legacy_sig2ux_matrix(sigma_y) @ sigma
        U = ux[: self.n_U]
        X = None if self.n_X == 0 else ux[self.n_U :]
        return U, X

    def ux2sig(self, U: np.ndarray, X: np.ndarray | None, sigma_y: float) -> np.ndarray:
        """Legacy-compatible ``(U, X)`` -> stress conversion."""
        U = _as_float_vector("U", U, self.n_U)
        if self.n_X == 0:
            if X is not None:
                x_arr = np.asarray(X, dtype=np.float64).reshape(-1)
                if x_arr.size != 0:
                    raise ValueError("CPS4 does not use an X variable.")
            ux = U
        else:
            if X is None:
                raise ValueError(f"{self.ele_type} requires X with length {self.n_X}.")
            X = _as_float_vector("X", X, self.n_X)
            ux = np.concatenate([U, X], axis=0)
        return self.get_legacy_ux2sig_matrix(sigma_y) @ ux

    def convert_c_matrix_block_legacy(
        self,
        c_block: np.ndarray,
        sigma_y: float,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Convert one stress block of ``C`` to the legacy MN blocks.

        Parameters
        ----------
        c_block
            Stress block with shape ``(n_eq, n_str)`` for one Gauss point.
        sigma_y
            Yield strength at that Gauss point.

        Returns
        -------
        A_block, B_block
            Exactly matches the legacy block meaning:
            - 3D/CPE: ``[A_block, B_block] == c_block @ UX2Sig(sigma_y)``
            - CPS:    ``A_block == c_block @ UX2Sig(sigma_y)``, ``B_block is None``
        """
        c_block = np.asarray(c_block, dtype=np.float64)
        if c_block.ndim != 2 or c_block.shape[1] != self.n_str:
            raise ValueError(
                f"c_block must have shape (n_eq, {self.n_str}), got {c_block.shape}."
            )

        mn_block = c_block @ self.get_legacy_ux2sig_matrix(sigma_y)
        A_block = mn_block[:, : self.n_U]
        B_block = None if self.n_X == 0 else mn_block[:, self.n_U :]
        return A_block, B_block

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def build_A_mat(self, sigma_y: float) -> np.ndarray:
        """Compatibility wrapper for the legacy stress-to-U block."""
        return self.get_legacy_sig2ux_matrix(sigma_y)[: self.n_U, :]

    def build_B_vec(self, sigma_y: float) -> np.ndarray | None:
        """Compatibility wrapper for the legacy stress-to-X block."""
        if self.n_X == 0:
            return None
        return self.get_legacy_sig2ux_matrix(sigma_y)[self.n_U :, :]
