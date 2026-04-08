"""
jaxmech.model.result - Unified analysis result container.

The AnalysisResult is the standard output of any FEM analysis (elastic,
nonlinear, etc.) and serves as the input to higher-level modules
(shakedown, topology optimization, direct methods, etc.).

Design notes:
  - `operators` stores assembled matrices (C_sparse, G_int, etc.) that
    downstream modules need. This avoids re-assembling per module.
  - All array fields use JAX arrays where possible, but numpy is also
    accepted for interoperability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np


def _csr_to_triplet_dict(mat, prefix: str) -> dict[str, np.ndarray]:
    from scipy.sparse import coo_matrix

    coo = coo_matrix(mat)
    return {
        f"{prefix}_shape": np.asarray(coo.shape, dtype=np.int64),
        f"{prefix}_row": np.asarray(coo.row, dtype=np.int64),
        f"{prefix}_col": np.asarray(coo.col, dtype=np.int64),
        f"{prefix}_data": np.asarray(coo.data, dtype=np.float64),
    }


def _triplet_dict_to_csr(data: dict, prefix: str):
    from scipy.sparse import coo_matrix

    shape = tuple(np.asarray(data[f"{prefix}_shape"]).reshape(-1).tolist())
    row = np.asarray(data[f"{prefix}_row"], dtype=np.int64).reshape(-1)
    col = np.asarray(data[f"{prefix}_col"], dtype=np.int64).reshape(-1)
    val = np.asarray(data[f"{prefix}_data"], dtype=np.float64).reshape(-1)
    return coo_matrix((val, (row, col)), shape=shape, dtype=np.float64).tocsr()


@dataclass
class Operators:
    """Assembled FEM operators shared by downstream modules.

    Attributes:
        C_sparse:       Equilibrium matrix (stress -> nodal force),
                        shape (n_free_dofs, n_gauss_total * n_str).
        C_gen:          Generalized equilibrium matrix (for shell),
                        shape (n_free_dofs, n_gauss_total * n_gen_str).
        G_int:          Layer-to-generalized integration operator (for shell),
                        mapping layer stresses to generalized forces.
        B_ext:          External force operator (optional).
        K_global:       Global stiffness matrix (optional, may be large).
        free_dofs:      Array of free DOF indices.
        n_str:          Number of stress components per Gauss point (3 or 6).
    """
    C_sparse: Optional[Any] = None
    C_gen: Optional[Any] = None
    G_int: Optional[Any] = None
    B_ext: Optional[Any] = None
    K_global: Optional[Any] = None
    free_dofs: Optional[np.ndarray] = None
    n_str: int = 6


@dataclass
class ShellRecoveryResult:
    """Shell-specific recovered fields not shared by all analysis families.

    Attributes:
        u_nodal:             Nodal shell DOFs reshaped as ``(n_nodes, 6)``.
        gen_internal_force:  Generalized internal nodal forces with shell DOF
                             layout, shape ``(n_nodes, 6)``.
        layer_strain:        Through-thickness strains,
                             shape ``(n_elem, n_gp, n_section_points, 3)``.
        layer_stress:        Through-thickness stresses,
                             shape ``(n_elem, n_gp, n_section_points, 3)``.
        section_z:           Normalized section-point coordinates through
                             thickness, typically on [-0.5, 0.5].
        elem_thickness:      Per-element shell thickness array.
    """
    u_nodal: Optional[np.ndarray] = None
    gen_internal_force: Optional[np.ndarray] = None
    layer_strain: Optional[np.ndarray] = None
    layer_stress: Optional[np.ndarray] = None
    section_z: Optional[np.ndarray] = None
    elem_thickness: Optional[np.ndarray] = None


@dataclass
class AnalysisResult:
    """Unified result container for a single-load-case FEM analysis.

    Attributes:
        u:              Global displacement vector or primary solved DOF array.
        reaction:       Reaction force array (optional), same shape as u.
        gauss_strain:   Gauss point strains, shape ``(n_elements, n_gp, n_str)``
                for uniform meshes or flat ``(n_gauss_total, n_str)``
                for mixed meshes.
        gauss_stress:   Gauss point stresses, same shape convention.
        internal_force: Nodal internal force vector (optional).
        gauss_coords:   Gauss point physical coordinates (optional).
        gauss_vols:     Gauss point volumes/weights (optional).
        n_gauss_per_elem: Number of Gauss points per element in mesh order.
        operators:      Assembled FEM operators for downstream modules.
        ele_yield:      Per-Gauss-point yield stress (for shakedown input).
        shell:          Shell-only recovery payload. Present only for shell
                        analyses.
        metadata:       Free-form metadata dict. Suggested keys:
                        - "load_case_name"
                        - "analysis_type": "linear_elastic", "nonlinear", ...
                        - "solver_info"
                        - "validated_with_ODB": bool
    """
    u: Optional[np.ndarray] = None
    reaction: Optional[np.ndarray] = None
    gauss_strain: Optional[np.ndarray] = None
    gauss_stress: Optional[np.ndarray] = None
    gauss_peeq: Optional[np.ndarray] = None
    internal_force: Optional[np.ndarray] = None
    gauss_coords: Optional[np.ndarray] = None
    gauss_vols: Optional[np.ndarray] = None
    n_gauss_per_elem: Optional[np.ndarray] = None
    operators: Operators = field(default_factory=Operators)
    ele_yield: Optional[np.ndarray] = None
    shell: Optional[ShellRecoveryResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_mat(self, filepath: str) -> None:
        """Export analysis result to MATLAB .mat format.

        This method serializes the generic analysis result itself.
        For shakedown-specific MAT bundles, use
        ``jaxmech.modules.shakedown.export.export_shakedown_mat(...)`` so the
        yield field and operator layout stay owned by the shakedown layer.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .mat).

        Notes
        -----
        Exported fields:
          - 'u': Nodal displacements (n_nodes, n_dof_per_node)
          - 'gauss_stress': Gauss stresses, uniform or flat mixed layout
          - 'gauss_strain': Gauss strains, uniform or flat mixed layout
          - 'n_gauss_per_elem': Per-element Gauss point counts for mixed meshes
          - 'internal_force': Internal nodal forces (n_nodes*n_dof_per_node,)
          - 'reaction': Reaction forces at prescribed DOFs (if available)
          - 'ele_yield': Optional auxiliary per-Gauss yield field
          - 'shell_*': Optional shell-only recovered fields
          - 'metadata': Analysis metadata as a nested dict

        Examples
        --------
        >>> result = run_analysis(model)
        >>> result.to_mat("analysis_output.mat")
        """
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError(
                "scipy is required for .mat export. Install with: pip install scipy"
            )

        mat_dict = {}

        # Primary result fields
        if self.u is not None:
            mat_dict["u"] = np.asarray(self.u)

        if self.gauss_stress is not None:
            mat_dict["gauss_stress"] = np.asarray(self.gauss_stress)

        if self.gauss_peeq is not None:
            mat_dict["gauss_peeq"] = np.asarray(self.gauss_peeq)

        if self.gauss_strain is not None:
            mat_dict["gauss_strain"] = np.asarray(self.gauss_strain)

        if self.internal_force is not None:
            mat_dict["internal_force"] = np.asarray(self.internal_force)

        # Optional fields
        if self.reaction is not None:
            mat_dict["reaction"] = np.asarray(self.reaction)

        if self.ele_yield is not None:
            mat_dict["ele_yield"] = np.asarray(self.ele_yield)

        if self.gauss_coords is not None:
            mat_dict["gauss_coords"] = np.asarray(self.gauss_coords)

        if self.gauss_vols is not None:
            mat_dict["gauss_vols"] = np.asarray(self.gauss_vols)

        if self.n_gauss_per_elem is not None:
            mat_dict["n_gauss_per_elem"] = np.asarray(self.n_gauss_per_elem)

        if self.operators.C_sparse is not None:
            mat_dict.update(_csr_to_triplet_dict(self.operators.C_sparse, "C_sparse"))

        if self.operators.C_gen is not None:
            mat_dict["C_gen"] = self.operators.C_gen

        if self.operators.G_int is not None:
            mat_dict.update(_csr_to_triplet_dict(self.operators.G_int, "G_int"))

        if self.shell is not None:
            if self.shell.u_nodal is not None:
                mat_dict["shell_u_nodal"] = np.asarray(self.shell.u_nodal)
            if self.shell.gen_internal_force is not None:
                mat_dict["shell_gen_internal_force"] = np.asarray(self.shell.gen_internal_force)
            if self.shell.layer_strain is not None:
                mat_dict["shell_layer_strain"] = np.asarray(self.shell.layer_strain)
            if self.shell.layer_stress is not None:
                mat_dict["shell_layer_stress"] = np.asarray(self.shell.layer_stress)
            if self.shell.section_z is not None:
                mat_dict["shell_section_z"] = np.asarray(self.shell.section_z)
            if self.shell.elem_thickness is not None:
                mat_dict["shell_elem_thickness"] = np.asarray(self.shell.elem_thickness)

        # Metadata (convert to string for compatibility)
        if self.metadata:
            mat_dict["metadata"] = {
                k: str(v) for k, v in self.metadata.items()
            }

        # Save to file
        sio.savemat(filepath, mat_dict, oned_as="column")

    @classmethod
    def from_mat(cls, filepath: str) -> "AnalysisResult":
        """Load analysis result from MATLAB .mat format.

        Parameters
        ----------
        filepath : str
            Input .mat file path.

        Returns
        -------
        AnalysisResult
            Restored analysis result.

        Examples
        --------
        >>> result = AnalysisResult.from_mat("analysis_output.mat")
        """
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError(
                "scipy is required for .mat import. Install with: pip install scipy"
            )

        mat_data = sio.loadmat(filepath, squeeze_me=True)

        # Extract arrays
        u = mat_data.get("u")
        solid_u_nodal = mat_data.get("solid_u_nodal")
        solid_nforc_nodal = mat_data.get("solid_nforc_nodal")
        gauss_stress = mat_data.get("gauss_stress")
        gauss_peeq = mat_data.get("gauss_peeq")
        gauss_strain = mat_data.get("gauss_strain")
        internal_force = mat_data.get("internal_force")
        reaction = mat_data.get("reaction")
        ele_yield = mat_data.get("ele_yield")
        gauss_coords = mat_data.get("gauss_coords")
        gauss_vols = mat_data.get("gauss_vols")
        n_gauss_per_elem = mat_data.get("n_gauss_per_elem")
        c_sparse = None
        if all(key in mat_data for key in ("C_sparse_shape", "C_sparse_row", "C_sparse_col", "C_sparse_data")):
            c_sparse = _triplet_dict_to_csr(mat_data, "C_sparse")
        elif "C_sparse" in mat_data:
            c_sparse = mat_data.get("C_sparse")
        c_gen = mat_data.get("C_gen")

        g_int = None
        if all(key in mat_data for key in ("G_int_shape", "G_int_row", "G_int_col", "G_int_data")):
            g_int = _triplet_dict_to_csr(mat_data, "G_int")
        elif "G_int" in mat_data:
            g_int = mat_data.get("G_int")
        shell_u_nodal = mat_data.get("shell_u_nodal")
        shell_gen_internal_force = mat_data.get("shell_gen_internal_force")
        shell_layer_strain = mat_data.get("shell_layer_strain")
        shell_layer_stress = mat_data.get("shell_layer_stress")
        shell_section_z = mat_data.get("shell_section_z")
        shell_elem_thickness = mat_data.get("shell_elem_thickness")

        # Extract metadata (may be string, dict-like, or MATLAB struct array)
        metadata = {}
        if "metadata" in mat_data:
            meta_raw = mat_data["metadata"]
            if isinstance(meta_raw, dict):
                metadata = {k: v.item() if hasattr(v, "item") else v
                            for k, v in meta_raw.items()}
            elif hasattr(meta_raw, "dtype") and getattr(meta_raw.dtype, "names", None):
                metadata = {
                    name: meta_raw[name].item() if hasattr(meta_raw[name], "item") else meta_raw[name]
                    for name in meta_raw.dtype.names
                }

        shell = None
        if any(
            val is not None
            for val in (
                shell_u_nodal,
                shell_gen_internal_force,
                shell_layer_strain,
                shell_layer_stress,
                shell_section_z,
                shell_elem_thickness,
            )
        ):
            shell = ShellRecoveryResult(
                u_nodal=shell_u_nodal,
                gen_internal_force=shell_gen_internal_force,
                layer_strain=shell_layer_strain,
                layer_stress=shell_layer_stress,
                section_z=shell_section_z,
                elem_thickness=shell_elem_thickness,
            )

        if u is None and shell_u_nodal is not None:
            u = np.asarray(shell_u_nodal, dtype=np.float64).reshape(-1)
        if u is None and solid_u_nodal is not None:
            u = np.asarray(solid_u_nodal, dtype=np.float64).reshape(-1)
        if internal_force is None and solid_nforc_nodal is not None:
            internal_force = -np.asarray(solid_nforc_nodal, dtype=np.float64).reshape(-1)

        return cls(
            u=u,
            gauss_strain=gauss_strain,
            gauss_stress=gauss_stress,
            gauss_peeq=gauss_peeq,
            internal_force=internal_force,
            reaction=reaction,
            gauss_coords=gauss_coords,
            gauss_vols=gauss_vols,
            n_gauss_per_elem=n_gauss_per_elem,
            operators=Operators(C_sparse=c_sparse, C_gen=c_gen, G_int=g_int),
            ele_yield=ele_yield,
            shell=shell,
            metadata=metadata,
        )
