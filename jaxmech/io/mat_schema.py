"""Shared MAT schema helpers for compact jaxmech result files."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

FRAME_OUTPUTS_KEY = "frame_outputs"
SOLID_PARAM_KEY = "solid_param"
SHELL_PARAM_KEY = "shell_param"
VALIDATION_RESULT_KEY = "ValidationResult"
VALIDATION_PARAM_KEY = "validation_param"

FRAME_OUTPUT_PREFIXES = ("frame_", "abaqus_frame_")

SHELL_PARAM_FIELDS = (
    "n_str",
    "n_gauss_per_elem",
    "shell_section_z",
    "shell_simpson_weights",
    "shell_elem_thickness",
    "shell_surface_section_indices",
    "shell_surface_names",
    "n_generalized_str",
)

SOLID_PARAM_FIELDS = (
    "n_str",
    "n_gauss_per_elem",
    "solid_elem_types",
    "free_dofs",
    "ndof_per_node",
    "voigt_order",
    "gauss_layout",
)

VALIDATION_RESULT_FIELDS = (
    ("validated_with_ODB", "validated_with_ODB"),
    ("nonlinear_validation_with_ODB", "nonlinear_validation_with_ODB"),
    ("shell_full_model_validation", "shell_full_model_validation"),
    ("validated_ODBName", "validated_ODBName"),
    ("validation_frame_time", "frame_time"),
    ("abaqus_validation_frame_time", "abaqus_frame_time"),
    ("validation_abaqus_frame_indices", "abaqus_frame_indices"),
    ("validation_source_frame_time", "source_frame_time"),
    ("abaqus_nforc_available", "abaqus_nforc_available"),
    ("Error_with_ODB", "error_summary"),
    ("ErrorHistory_with_ODB", "error_history"),
)

VALIDATION_PARAM_FIELDS = (
    ("validation_gauss_permutation_jax_to_abaqus", "gauss_permutation_jax_to_abaqus"),
    ("validation_component_transform_name", "component_transform_name"),
    ("validation_component_permutation_jax_to_abaqus", "component_permutation_jax_to_abaqus"),
    ("validation_component_sign_jax_to_abaqus", "component_sign_jax_to_abaqus"),
    ("validation_gen_component_permutation_jax_to_abaqus", "gen_component_permutation_jax_to_abaqus"),
    ("validation_gen_component_sign_jax_to_abaqus", "gen_component_sign_jax_to_abaqus"),
    ("validation_shell_basis_rotation_jax_to_abaqus", "shell_basis_rotation_jax_to_abaqus"),
    ("validation_shell_basis_angles_deg", "shell_basis_angles_deg"),
    ("validation_section_indices", "section_indices"),
    ("validation_abaqus_surface_section_indices", "abaqus_surface_section_indices"),
    ("validation_abaqus_source_surface_section_indices", "abaqus_source_surface_section_indices"),
    ("validation_abaqus_surface_section_numbers", "abaqus_surface_section_numbers"),
    ("validation_abaqus_all_section_numbers", "abaqus_all_section_numbers"),
    ("validation_abaqus_surface_section_names", "abaqus_surface_section_names"),
    ("validation_jax_surface_section_indices", "jax_surface_section_indices"),
    ("validation_abaqus_section_count", "abaqus_section_count"),
    ("validation_abaqus_all_section_count", "abaqus_all_section_count"),
    ("validation_jax_section_count", "jax_section_count"),
    ("validation_voigt_perm", "voigt_perm"),
    ("validation_ip_permutation_jax_to_abaqus", "ip_permutation_jax_to_abaqus"),
    ("validation_abaqus_ip_order", "abaqus_ip_order"),
)


def _unwrap_scalar(value: Any) -> Any:
    cur = value
    for _ in range(8):
        if isinstance(cur, np.ndarray) and cur.size == 1 and not getattr(cur.dtype, "names", None):
            cur = cur.reshape(-1)[0]
            continue
        break
    return cur


def mat_value_to_plain(value: Any) -> Any:
    """Convert scipy-loaded scalar structs/cells to plain Python containers."""
    cur = _unwrap_scalar(value)
    if isinstance(cur, np.ndarray) and cur.dtype.kind in {"U", "S"}:
        if cur.ndim == 0:
            cur = cur.item()
        elif cur.size == 1:
            cur = cur.reshape(-1)[0]
        else:
            return "".join(str(item) for item in cur.reshape(-1)).strip()
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore")
    if isinstance(cur, Mapping):
        return {str(key): mat_value_to_plain(val) for key, val in cur.items()}
    if hasattr(cur, "_fieldnames"):
        return {str(name): mat_value_to_plain(getattr(cur, name)) for name in cur._fieldnames}
    if isinstance(cur, np.ndarray) and getattr(cur.dtype, "names", None) and cur.size == 1:
        rec = cur.reshape(-1)[0]
        return {str(name): mat_value_to_plain(rec[name]) for name in cur.dtype.names or ()}
    return value


def _container_dict(value: Any) -> dict[str, Any]:
    plain = mat_value_to_plain(value)
    if isinstance(plain, Mapping):
        return {str(key): val for key, val in plain.items()}
    return {}


def _text_or_value(value: Any) -> Any:
    cur = mat_value_to_plain(value)
    for _ in range(4):
        if isinstance(cur, (list, tuple)) and len(cur) == 1:
            cur = cur[0]
            continue
        if isinstance(cur, np.ndarray) and cur.size == 1:
            cur = cur.reshape(-1)[0]
            continue
        break
    if isinstance(cur, bytes):
        return cur.decode("utf-8", errors="ignore")
    if isinstance(cur, np.generic):
        return cur.item()
    return cur


def _clean_metadata(value: Any) -> Any:
    data = _container_dict(value)
    if not data:
        return value
    return {str(key): _text_or_value(val) for key, val in data.items()}


def _is_shell_payload(payload: Mapping[str, Any]) -> bool:
    metadata = _clean_metadata(payload.get("metadata", {}))
    if isinstance(metadata, Mapping) and str(metadata.get("family", "")).strip().lower() == "shell":
        return True
    if payload.get(SHELL_PARAM_KEY) is not None:
        return True
    return any(str(key).startswith("shell_") or "shell_layer" in str(key) for key in payload.keys())


def _is_solid_payload(payload: Mapping[str, Any]) -> bool:
    metadata = _clean_metadata(payload.get("metadata", {}))
    if isinstance(metadata, Mapping):
        family = str(metadata.get("family", "")).strip().lower()
        if family == "shell":
            return False
        if family == "solid":
            return True
    if payload.get(SOLID_PARAM_KEY) is not None:
        return True
    if payload.get(SHELL_PARAM_KEY) is not None:
        return False
    if any(str(key).startswith("shell_") or "shell_layer" in str(key) for key in payload.keys()):
        return False
    return "InpData" in payload and any(key in payload for key in ("frame_gauss_stress", "gauss_stress"))


def _clean_state_layout(layout: Any) -> Any:
    data = _container_dict(layout)
    if not data:
        return layout
    data.pop("component_names", None)
    try:
        n_str = int(np.asarray(data.get("n_str", 0)).reshape(-1)[0])
    except Exception:
        n_str = 0
    try:
        n_state = int(np.asarray(data.get("n_state_vars", 0)).reshape(-1)[0])
    except Exception:
        n_state = 0
    family = str(np.asarray(data.get("family", "")).reshape(-1)[0] if np.asarray(data.get("family", "")).size else data.get("family", "")).lower()
    if "plastic_strain_start_index" not in data:
        data["plastic_strain_start_index"] = 1
    if "plastic_strain_count" not in data:
        data["plastic_strain_count"] = 4 if family == "j2_plane_stress" else n_str
    if "eqps_index" not in data:
        data["eqps_index"] = 5 if family == "j2_plane_stress" else n_str + 1
    if family == "j2_plane_stress":
        data.setdefault("plane_stress_pe33_index", 6)
        data.setdefault("abaqus_pe33_output_index", 3)
    elif n_state > n_str + 1 and "back_stress_start_index" not in data:
        data["back_stress_start_index"] = n_str + 2
        data["back_stress_count"] = n_str
    return data


def clean_public_structs(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Remove accidental scalar cell wrappers from public top-level structs."""
    for key in (
        "InpData",
        "metadata",
        "PlasticResult",
        "PlasticStateLayout",
        "plastic_state_layout",
        VALIDATION_RESULT_KEY,
        VALIDATION_PARAM_KEY,
    ):
        if key in payload:
            payload[key] = mat_value_to_plain(payload[key])
    if "metadata" in payload:
        payload["metadata"] = _clean_metadata(payload["metadata"])
    legacy_layout = None
    if "PlasticStateLayout" in payload:
        legacy_layout = payload.pop("PlasticStateLayout")
    if "plastic_state_layout" in payload:
        legacy_layout = payload.pop("plastic_state_layout")
    plastic_result = _container_dict(payload.get("PlasticResult"))
    if legacy_layout is not None and "state_layout" not in plastic_result:
        plastic_result["state_layout"] = legacy_layout
    if plastic_result:
        if "state_layout" in plastic_result:
            plastic_result["state_layout"] = _clean_state_layout(plastic_result["state_layout"])
        payload["PlasticResult"] = plastic_result
    if SHELL_PARAM_KEY in payload:
        payload[SHELL_PARAM_KEY] = _container_dict(payload[SHELL_PARAM_KEY])
    if SOLID_PARAM_KEY in payload:
        payload[SOLID_PARAM_KEY] = _container_dict(payload[SOLID_PARAM_KEY])
    if FRAME_OUTPUTS_KEY in payload:
        payload[FRAME_OUTPUTS_KEY] = _container_dict(payload[FRAME_OUTPUTS_KEY])
    return payload


def pack_shell_param(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Move shell-specific parameters into ``shell_param``."""
    if not _is_shell_payload(payload):
        payload.pop(SHELL_PARAM_KEY, None)
        return payload
    shell_param = _container_dict(payload.get(SHELL_PARAM_KEY))
    for key in SHELL_PARAM_FIELDS:
        if key in payload:
            shell_param[key] = payload.pop(key)
    if shell_param:
        payload[SHELL_PARAM_KEY] = shell_param
    return payload


def expand_shell_param(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Expose ``shell_param`` fields in a transient flat runtime view."""
    shell_param = _container_dict(payload.get(SHELL_PARAM_KEY))
    for key, value in shell_param.items():
        payload.setdefault(str(key), value)
    return payload


def pack_solid_param(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Move solid-specific layout parameters into ``solid_param``."""
    if not _is_solid_payload(payload):
        payload.pop(SOLID_PARAM_KEY, None)
        return payload
    solid_param = _container_dict(payload.get(SOLID_PARAM_KEY))
    for key in SOLID_PARAM_FIELDS:
        if key in payload:
            solid_param[key] = payload.pop(key)
    if solid_param:
        payload[SOLID_PARAM_KEY] = solid_param
    return payload


def expand_solid_param(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Expose ``solid_param`` fields in a transient flat runtime view."""
    solid_param = _container_dict(payload.get(SOLID_PARAM_KEY))
    for key, value in solid_param.items():
        payload.setdefault(str(key), value)
    return payload


def pack_frame_outputs(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Move all frame history arrays into ``frame_outputs``."""
    frame_outputs = _container_dict(payload.get(FRAME_OUTPUTS_KEY))
    for key in list(payload.keys()):
        if key == FRAME_OUTPUTS_KEY:
            continue
        if any(str(key).startswith(prefix) for prefix in FRAME_OUTPUT_PREFIXES):
            frame_outputs[str(key)] = payload.pop(key)
    if frame_outputs:
        payload[FRAME_OUTPUTS_KEY] = frame_outputs
    return payload


def expand_frame_outputs(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Expose ``frame_outputs`` fields in a transient flat runtime view."""
    frame_outputs = _container_dict(payload.get(FRAME_OUTPUTS_KEY))
    for key, value in frame_outputs.items():
        payload.setdefault(str(key), value)
    return payload


def expand_derived_frame_fields(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Expose derived frame fields needed by legacy runtime helpers.

    Public nonlinear MAT files store ``frame_nforc`` as the canonical nodal
    force due to stress.  Some validation helpers still consume the flat
    internal force vector; derive it in memory instead of saving both arrays.
    """
    if "frame_internal_force" not in payload and "frame_nforc" in payload:
        try:
            nforc = np.asarray(payload["frame_nforc"], dtype=np.float64)
            if nforc.ndim >= 3:
                payload["frame_internal_force"] = -nforc.reshape(nforc.shape[0], -1)
            else:
                payload["frame_internal_force"] = -nforc
        except Exception:
            pass
    return payload


def pack_validation_payload(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Move validation-only metadata/result fields into compact structs."""
    is_validation = (
        VALIDATION_RESULT_KEY in payload
        or bool(payload.get("validated_with_ODB") is not None)
        or any(str(key).startswith("abaqus_frame_") for key in payload.keys())
    )
    if not is_validation:
        payload.pop(VALIDATION_RESULT_KEY, None)
        payload.pop(VALIDATION_PARAM_KEY, None)
        return payload

    result = _container_dict(payload.get(VALIDATION_RESULT_KEY))
    for top_key, nested_key in VALIDATION_RESULT_FIELDS:
        if top_key in payload:
            result[nested_key] = payload.pop(top_key)
    if result:
        payload[VALIDATION_RESULT_KEY] = result

    param = _container_dict(payload.get(VALIDATION_PARAM_KEY))
    for top_key, nested_key in VALIDATION_PARAM_FIELDS:
        if top_key in payload:
            param[nested_key] = payload.pop(top_key)
    if param:
        payload[VALIDATION_PARAM_KEY] = param
    return payload


def expand_validation_payload(payload: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Expose validation structs in a transient flat runtime view."""
    result = _container_dict(payload.get(VALIDATION_RESULT_KEY))
    reverse_result = {nested: top for top, nested in VALIDATION_RESULT_FIELDS}
    for nested_key, value in result.items():
        payload.setdefault(reverse_result.get(str(nested_key), str(nested_key)), value)

    param = _container_dict(payload.get(VALIDATION_PARAM_KEY))
    reverse_param = {nested: top for top, nested in VALIDATION_PARAM_FIELDS}
    for nested_key, value in param.items():
        payload.setdefault(reverse_param.get(str(nested_key), str(nested_key)), value)
    return payload


def normalize_public_mat_payload(
    payload: Mapping[str, Any],
    *,
    pack_frames: bool = False,
    pack_solid: bool = True,
    pack_shell: bool = True,
    pack_validation: bool = False,
) -> dict[str, Any]:
    """Return a clean public MAT payload ready for ``scipy.io.savemat``."""
    out: dict[str, Any] = {str(key): val for key, val in payload.items() if not str(key).startswith("__")}
    clean_public_structs(out)
    if pack_solid:
        pack_solid_param(out)
    if pack_shell:
        pack_shell_param(out)
    if pack_validation:
        pack_validation_payload(out)
    if pack_frames:
        pack_frame_outputs(out)
    else:
        out.pop(FRAME_OUTPUTS_KEY, None)
    return out


def expand_public_mat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a flat runtime view for readers and comparison code."""
    out: dict[str, Any] = {str(key): val for key, val in payload.items() if not str(key).startswith("__")}
    clean_public_structs(out)
    expand_solid_param(out)
    expand_shell_param(out)
    expand_validation_payload(out)
    expand_frame_outputs(out)
    expand_derived_frame_fields(out)
    return out


def nested_frame_output_shapes(value: Any) -> dict[str, tuple[int, ...]]:
    """Return shapes of fields stored in a loaded ``frame_outputs`` struct."""
    return nested_struct_shapes(value)


def nested_struct_shapes(value: Any) -> dict[str, tuple[int, ...]]:
    """Return array shapes of fields stored in a loaded scalar MATLAB struct."""
    frame_outputs = _container_dict(value)
    shapes: dict[str, tuple[int, ...]] = {}
    for key, val in frame_outputs.items():
        try:
            shapes[str(key)] = tuple(int(v) for v in np.asarray(val).shape)
        except Exception:
            continue
    return shapes
