"""Derived field computations for visualisation.

Computes quantities like von Mises stress, principal stresses,
displacement magnitude from raw field arrays.
"""

from __future__ import annotations

import numpy as np

from jaxmech.modules.visualization.mat_reader import VizData, FieldInfo


def von_mises_6(s: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress from 6-component Voigt tensor.

    Components assumed: s11, s22, s33, s12, s13, s23.
    """
    s11, s22, s33 = s[..., 0], s[..., 1], s[..., 2]
    s12, s13, s23 = s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )


def von_mises_3(s: np.ndarray) -> np.ndarray:
    """Von Mises for 2-D plane stress (s11, s22, s12)."""
    s11, s22, s12 = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3.0 * s12 ** 2)


def displacement_magnitude(u: np.ndarray, n_nodes: int) -> np.ndarray:
    """Compute ||u|| per node from a flat or (n_nodes, ndof) displacement."""
    arr = np.asarray(u, dtype=np.float64)
    if arr.ndim == 1:
        ndof = arr.size // n_nodes if n_nodes > 0 else 3
        arr = arr.reshape(n_nodes, ndof)
    return np.linalg.norm(arr, axis=-1)


def principal_stresses_6(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute principal stresses from 6-component Voigt stress.

    Returns (sigma_1, sigma_2, sigma_3) each shape (...,).
    """
    s11, s22, s33 = s[..., 0], s[..., 1], s[..., 2]
    s12, s13, s23 = s[..., 3], s[..., 4], s[..., 5]

    shape = s11.shape
    flat = s11.size
    tensors = np.zeros((*shape, 3, 3), dtype=np.float64)
    tensors[..., 0, 0] = s11
    tensors[..., 1, 1] = s22
    tensors[..., 2, 2] = s33
    tensors[..., 0, 1] = s12
    tensors[..., 1, 0] = s12
    tensors[..., 0, 2] = s13
    tensors[..., 2, 0] = s13
    tensors[..., 1, 2] = s23
    tensors[..., 2, 1] = s23

    eigs = np.linalg.eigvalsh(tensors)  # sorted ascending
    return eigs[..., 2], eigs[..., 1], eigs[..., 0]


def hydrostatic_6(s: np.ndarray) -> np.ndarray:
    """Hydrostatic (mean) stress from 6-component Voigt."""
    return (s[..., 0] + s[..., 1] + s[..., 2]) / 3.0


def _is_shell_generalized_field(fi: FieldInfo, *, kind: str) -> bool:
    text = f"{fi.key} {fi.label} {fi.description}".lower()
    return (
        fi.n_components >= 6
        and "shell generalized" in text
        and kind in text
    )


def _register_scalar_split(
    vd: VizData,
    *,
    src_key: str,
    derived_key: str,
    component_slice: slice,
    label: str,
    symbol: str,
    formula: str,
    description: str,
) -> None:
    if src_key not in vd.fields or derived_key in vd.fields:
        return
    fi = vd.field_info.get(src_key)
    if fi is None or fi.n_components < component_slice.stop:
        return
    raw = np.asarray(vd.fields[src_key], dtype=np.float64)
    if raw.ndim < 1 or raw.shape[-1] < component_slice.stop:
        return
    derived = np.linalg.norm(raw[..., component_slice], axis=-1)
    if int(fi.n_frames) <= 1 and derived.ndim >= 2 and derived.shape[0] == 1:
        derived = derived[0]
    vd.fields[derived_key] = derived
    vd.field_info[derived_key] = FieldInfo(
        key=derived_key,
        label=label,
        n_components=1,
        location=fi.location,
        n_frames=fi.n_frames,
        symbol=symbol,
        formula=formula,
        description=description,
    )


def _register_shell_generalized_split_magnitudes(vd: VizData) -> None:
    """Expose separate membrane/curvature magnitudes for STRI3 shell fields."""
    stress_specs = {
        "gauss_stress": ("gauss_shell_sgen_sf_magnitude", "gauss_shell_sgen_sm_magnitude", ""),
        "shell_generalized_stress": ("shell_sgen_sf_magnitude", "shell_sgen_sm_magnitude", ""),
        "frame_gauss_stress": ("frame_shell_sgen_sf_magnitude", "frame_shell_sgen_sm_magnitude", " history"),
        "validation_jax_S": ("validation_jax_sgen_sf_magnitude", "validation_jax_sgen_sm_magnitude", " history"),
        "validation_abaqus_S": ("validation_abaqus_sgen_sf_magnitude", "validation_abaqus_sgen_sm_magnitude", " history"),
    }
    for src_key, (sf_key, sm_key, suffix) in stress_specs.items():
        fi = vd.field_info.get(src_key)
        if fi is None or not _is_shell_generalized_field(fi, kind="stress"):
            continue
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=sf_key,
            component_slice=slice(0, 3),
            label=f"SGEN SF magnitude{suffix}",
            symbol=r"\|\mathbf{SF}\|_2",
            formula=r"\|\mathbf{SF}\|_2=\sqrt{SF_{11}^2+SF_{22}^2+SF_{12}^2}",
            description="Magnitude of the shell membrane force-resultant components SF11/SF22/SF12.",
        )
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=sm_key,
            component_slice=slice(3, 6),
            label=f"SGEN SM magnitude{suffix}",
            symbol=r"\|\mathbf{SM}\|_2",
            formula=r"\|\mathbf{SM}\|_2=\sqrt{SM_{11}^2+SM_{22}^2+SM_{12}^2}",
            description="Magnitude of the shell bending moment-resultant components SM11/SM22/SM12.",
        )

    strain_specs = {
        "gauss_strain": ("gauss_shell_egen_ge_magnitude", "gauss_shell_egen_gk_magnitude", ""),
        "shell_generalized_strain": ("shell_egen_ge_magnitude", "shell_egen_gk_magnitude", ""),
        "frame_gauss_strain": ("frame_shell_egen_ge_magnitude", "frame_shell_egen_gk_magnitude", " history"),
        "validation_jax_E": ("validation_jax_egen_ge_magnitude", "validation_jax_egen_gk_magnitude", " history"),
        "validation_abaqus_E": ("validation_abaqus_egen_ge_magnitude", "validation_abaqus_egen_gk_magnitude", " history"),
    }
    for src_key, (ge_key, gk_key, suffix) in strain_specs.items():
        fi = vd.field_info.get(src_key)
        if fi is None or not _is_shell_generalized_field(fi, kind="strain"):
            continue
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=ge_key,
            component_slice=slice(0, 3),
            label=f"EGEN GE magnitude{suffix}",
            symbol=r"\|\mathbf{GE}\|_2",
            formula=r"\|\mathbf{GE}\|_2=\sqrt{GE_{11}^2+GE_{22}^2+GE_{12}^2}",
            description="Magnitude of the shell membrane strain components GE11/GE22/GE12.",
        )
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=gk_key,
            component_slice=slice(3, 6),
            label=f"EGEN GK magnitude{suffix}",
            symbol=r"\|\mathbf{GK}\|_2",
            formula=r"\|\mathbf{GK}\|_2=\sqrt{GK_{11}^2+GK_{22}^2+GK_{12}^2}",
            description="Magnitude of the shell curvature components GK11/GK22/GK12.",
        )


def _is_shell_node_vector_field(fi: FieldInfo, *, kind: str) -> bool:
    text = f"{fi.key} {fi.label} {fi.description}".lower()
    if fi.n_components < 6 or fi.location != "node":
        return False
    if kind == "u":
        return (
            "shell displacement" in text
            or "displacement/rotation" in text
            or fi.key in {"shell_u_nodal", "frame_u", "validation_jax_U", "validation_abaqus_U"}
        )
    if kind == "nforc":
        return (
            "shell generalized nodal force" in text
            or "force/moment" in text
            or fi.key in {"shell_gen_internal_force", "frame_nforc", "frame_shell_f_drill", "validation_jax_NFORC", "validation_abaqus_NFORC"}
        )
    return False


def _register_shell_node_vector_split_magnitudes(vd: VizData) -> None:
    """Expose separate translational/rotational and force/moment magnitudes."""
    u_specs = {
        "shell_u_nodal": ("shell_u_trans_magnitude", "shell_u_rot_magnitude", ""),
        "frame_u": ("frame_shell_u_trans_magnitude", "frame_shell_u_rot_magnitude", " history"),
        "validation_jax_U": ("validation_jax_u_trans_magnitude", "validation_jax_u_rot_magnitude", " history"),
        "validation_abaqus_U": ("validation_abaqus_u_trans_magnitude", "validation_abaqus_u_rot_magnitude", " history"),
    }
    for src_key, (u_key, ur_key, suffix) in u_specs.items():
        fi = vd.field_info.get(src_key)
        if fi is None or not _is_shell_node_vector_field(fi, kind="u"):
            continue
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=u_key,
            component_slice=slice(0, 3),
            label=f"U translational magnitude{suffix}",
            symbol=r"\|\mathbf{U}\|_2",
            formula=r"\|\mathbf{U}\|_2=\sqrt{U_1^2+U_2^2+U_3^2}",
            description="Magnitude of the shell translational displacement components U1/U2/U3.",
        )
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=ur_key,
            component_slice=slice(3, 6),
            label=f"UR rotational magnitude{suffix}",
            symbol=r"\|\mathbf{UR}\|_2",
            formula=r"\|\mathbf{UR}\|_2=\sqrt{UR_1^2+UR_2^2+UR_3^2}",
            description="Magnitude of the shell rotational degrees of freedom UR1/UR2/UR3.",
        )

    force_specs = {
        "shell_gen_internal_force": ("shell_nforc_force_magnitude", "shell_nforc_moment_magnitude", "NFORC", ""),
        "frame_nforc": ("frame_shell_nforc_force_magnitude", "frame_shell_nforc_moment_magnitude", "NFORC", " history"),
        "frame_shell_f_drill": ("frame_shell_f_drill_force_magnitude", "frame_shell_f_drill_moment_magnitude", "FDRILL", " history"),
        "validation_jax_NFORC": ("validation_jax_nforc_force_magnitude", "validation_jax_nforc_moment_magnitude", "NFORC", " history"),
        "validation_abaqus_NFORC": ("validation_abaqus_nforc_force_magnitude", "validation_abaqus_nforc_moment_magnitude", "NFORC", " history"),
        "rsdms_NFORC": ("rsdms_nforc_force_magnitude", "rsdms_nforc_moment_magnitude", "NFORC", " vertex history"),
        "rsdms_CEQ": ("rsdms_ceq_force_magnitude", "rsdms_ceq_moment_magnitude", "CEQ", " load-point history"),
        "rsdms_FERROR": ("rsdms_ferror_force_magnitude", "rsdms_ferror_moment_magnitude", "FERROR", " vertex history"),
        "shakedown_equality_violation": ("shakedown_ceq_force_magnitude", "shakedown_ceq_moment_magnitude", "CEQ", " history"),
    }
    for src_key, (force_key, moment_key, base_label, suffix) in force_specs.items():
        fi = vd.field_info.get(src_key)
        if fi is None:
            continue
        is_nforc = _is_shell_node_vector_field(fi, kind="nforc")
        is_shell_force_moment = fi.n_components >= 6 and fi.location == "node" and src_key in {
            "rsdms_NFORC",
            "rsdms_CEQ",
            "rsdms_FERROR",
            "shakedown_equality_violation",
        }
        if not (is_nforc or is_shell_force_moment):
            continue
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=force_key,
            component_slice=slice(0, 3),
            label=f"{base_label} force magnitude{suffix}",
            symbol=r"\|\mathbf{F}\|_2",
            formula=r"\|\mathbf{F}\|_2=\sqrt{F_1^2+F_2^2+F_3^2}",
            description=f"Magnitude of the shell {base_label} force components F1/F2/F3.",
        )
        _register_scalar_split(
            vd,
            src_key=src_key,
            derived_key=moment_key,
            component_slice=slice(3, 6),
            label=f"{base_label} moment magnitude{suffix}",
            symbol=r"\|\mathbf{M}\|_2",
            formula=r"\|\mathbf{M}\|_2=\sqrt{M_1^2+M_2^2+M_3^2}",
            description=f"Magnitude of the shell {base_label} moment components M1/M2/M3.",
        )


# ---------------------------------------------------------------------------
# Registration: add derived fields to VizData
# ---------------------------------------------------------------------------

def register_derived_fields(vd: VizData) -> None:
    """Scan existing fields in *vd* and add derived quantities.

    Called automatically after ``load_viz_data``.  Adds keys like
    ``"von_mises_stress"``, ``"displacement_magnitude"``, etc.
    """
    n_nodes = int(vd.points.shape[0]) if vd.points is not None else 0

    # Von Mises from gauss_stress or frame_gauss_stress
    for src_key, label_prefix, loc in [
        ("gauss_stress", "Mises Stress", "gauss"),
        ("elastic_gauss_stress", "Mises (elastic)", "gauss"),
    ]:
        if src_key not in vd.fields:
            continue
        raw = vd.fields[src_key]
        fi = vd.field_info[src_key]
        n_comp = fi.n_components
        vm_func = von_mises_6 if n_comp == 6 else (von_mises_3 if n_comp == 3 else None)
        if vm_func is None:
            continue
        derived_key = f"derived_vm_{src_key}"
        vd.fields[derived_key] = vm_func(raw)
        vd.field_info[derived_key] = FieldInfo(
            key=derived_key, label=label_prefix,
            n_components=1, location=loc, n_frames=1,
        )

    for src_key, label_prefix in [
        ("frame_gauss_stress", "Mises Stress (cycle)"),
    ]:
        if src_key not in vd.fields:
            continue
        raw = vd.fields[src_key]
        fi = vd.field_info[src_key]
        n_comp = fi.n_components
        vm_func = von_mises_6 if n_comp == 6 else (von_mises_3 if n_comp == 3 else None)
        if vm_func is None:
            continue
        derived_key = f"derived_vm_{src_key}"
        n_frames = raw.shape[0]
        result = np.stack([vm_func(raw[i]) for i in range(n_frames)])
        vd.fields[derived_key] = result
        vd.field_info[derived_key] = FieldInfo(
            key=derived_key, label=label_prefix,
            n_components=1, location=fi.location, n_frames=n_frames,
        )

    # RSDM cycle von Mises
    for src_key, label_prefix in [
        ("residual_stress_cycle", "Mises Residual (cycle)"),
        ("total_stress_cycle", "Mises Total (cycle)"),
        ("excess_stress_cycle", "Mises Excess (cycle)"),
    ]:
        if src_key not in vd.fields:
            continue
        raw = vd.fields[src_key]
        fi = vd.field_info[src_key]
        n_comp = fi.n_components
        vm_func = von_mises_6 if n_comp == 6 else (von_mises_3 if n_comp == 3 else None)
        if vm_func is None:
            continue
        derived_key = f"derived_vm_{src_key}"
        n_frames = raw.shape[0]
        result = np.stack([vm_func(raw[i]) for i in range(n_frames)])
        vd.fields[derived_key] = result
        vd.field_info[derived_key] = FieldInfo(
            key=derived_key, label=label_prefix,
            n_components=1, location=fi.location, n_frames=n_frames,
        )

    # Displacement magnitude
    for src_key, label in [("u", "Displacement Magnitude"), ("elastic_u", "Elastic Disp. Mag.")]:
        if src_key not in vd.fields:
            continue
        raw = vd.fields[src_key]
        if n_nodes <= 0:
            continue
        derived_key = f"derived_mag_{src_key}"
        vd.fields[derived_key] = displacement_magnitude(raw, n_nodes)
        vd.field_info[derived_key] = FieldInfo(
            key=derived_key, label=label,
            n_components=1, location="node", n_frames=1,
        )

    # Frame displacement magnitude
    if "frame_u" in vd.fields and n_nodes > 0:
        raw = vd.fields["frame_u"]
        n_frames = raw.shape[0]
        mags = np.stack([displacement_magnitude(raw[i], n_nodes) for i in range(n_frames)])
        derived_key = "derived_mag_frame_u"
        vd.fields[derived_key] = mags
        vd.field_info[derived_key] = FieldInfo(
            key=derived_key, label="Disp. Magnitude (cycle)",
            n_components=1, location="node", n_frames=n_frames,
        )

    _register_shell_generalized_split_magnitudes(vd)
    _register_shell_node_vector_split_magnitudes(vd)
