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
