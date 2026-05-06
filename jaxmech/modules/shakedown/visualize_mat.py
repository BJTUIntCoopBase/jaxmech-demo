"""Visualization manifest builders for shakedown MAT files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jaxmech.model.viz_manifest import build_viz_manifest, field_descriptor


def _has(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if value is None:
        return False
    if isinstance(value, list):
        return any(_has({"value": item}, "value") for item in value)
    try:
        return np.asarray(value).size > 0
    except Exception:
        return False


def _lookup_in_obj(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping) and key in obj:
        return obj.get(key)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        values = []
        for item in obj.flat:
            value = _lookup_in_obj(item, key)
            if value is not None:
                values.append(value)
        if not values:
            return None
        return values[0] if len(values) == 1 else values
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None) and key in obj.dtype.names:
        try:
            src = obj.flat[0] if getattr(obj, "size", 0) == 1 else obj
            return src[key]
        except Exception:
            pass
    if hasattr(obj, key):
        return getattr(obj, key)
    return None


def _lookup(payload: Mapping[str, Any], key: str) -> Any:
    if key in payload:
        return payload.get(key)
    for parent in ("ElasticInputSet", "ResultsSet", "ConfigInfo"):
        value = _lookup_in_obj(payload.get(parent), key)
        if value is not None:
            return value
    return None


def _lookup_text(payload: Mapping[str, Any], key: str, default: str = "") -> str:
    value = _lookup(payload, key)
    if value is None:
        return default
    cur = value
    for _ in range(8):
        if isinstance(cur, np.ndarray):
            if cur.dtype.kind in {"U", "S"}:
                if cur.size == 1:
                    cur = cur.reshape(-1)[0]
                    continue
                return "".join(str(item) for item in cur.reshape(-1)).strip()
            if cur.dtype == object and cur.size == 1:
                cur = cur.reshape(-1)[0]
                continue
        break
    return str(cur).strip() if cur is not None else default


def _has_any(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_lookup(payload, key) is not None for key in keys)


def _n_str(payload: Mapping[str, Any]) -> int:
    for key in ("residual_stress", "sigma_E", "gauss_stress_cases"):
        value = _lookup(payload, key)
        if isinstance(value, list):
            value = next((item for item in value if item is not None), None)
        if value is None:
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.ndim >= 2 and int(arr.shape[-1]) in (3, 6):
            return int(arr.shape[-1])
        if arr.ndim >= 2 and key == "gauss_stress_cases" and int(arr.shape[1]) in (3, 6):
            return int(arr.shape[1])
    return 6


def build_shakedown_viz_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Declare fields derived from shakedown solver MAT contents."""
    fields: list[dict[str, Any]] = []
    n_comp = _n_str(payload)
    family = _lookup_text(payload, "family", "solid").lower() or "solid"
    shell_yield = _lookup_text(payload, "shell_yield", "").lower()
    residual_kind = _lookup_text(payload, "residual_stress_kind", "").lower()
    is_shell = family == "shell" or _has_any(payload, ("shell_layer_stress_cases", "C_gen", "gauss_thickness"))
    is_layer = is_shell and (shell_yield == "layer" or residual_kind == "layer" or _has_any(payload, ("shell_layer_stress_cases",)))
    is_ilyushin = is_shell and not is_layer

    if _has_any(payload, ("residual_stress",)) and not is_layer and not is_ilyushin:
        fields.append(
            field_descriptor(
                "shakedown_residual_stress",
                label="Shakedown residual stress",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                frames="single",
                default_selected=True,
                source_key="residual_stress",
                scalar_options=["components", "mises"],
                symbol=r"\boldsymbol{\rho}",
                formula=r"\boldsymbol{\rho}=\boldsymbol{\sigma}^{tot}-\alpha\boldsymbol{\sigma}^{E}",
                description="Self-equilibrated residual stress from the shakedown optimization",
            )
        )
    if _has_any(payload, ("residual_stress",)) and _has_any(payload, ("gauss_stress_cases", "sigma_E")) and not is_layer and not is_ilyushin:
        fields.append(
            field_descriptor(
                "shakedown_total_stress",
                label="Shakedown total stress",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_keys=["residual_stress", "gauss_stress_cases", "sigma_E", "vertex_load_matrix", "objective_value"],
                scalar_options=["components", "mises"],
                symbol=r"\boldsymbol{\sigma}^{tot}",
                formula=r"\boldsymbol{\sigma}^{tot}_v=\boldsymbol{\rho}+\alpha\boldsymbol{\sigma}^{E}_v",
                description="Total stress at each load-domain vertex",
            )
        )
    if is_layer and _has_any(payload, ("residual_stress", "shell_layer_stress_cases")):
        for surface, label_suffix in (("SNEG", "inner/SNEG"), ("SPOS", "outer/SPOS")):
            fields.append(
                field_descriptor(
                    f"shakedown_S_res_{surface}",
                    label=f"Shakedown S residual ({label_suffix})",
                    family="stress",
                    location="gauss",
                    tensor_kind="stress_voigt",
                    n_components=3,
                    frames="frames",
                    frame_axis="load_point",
                    default_selected=surface == "SPOS",
                    source_keys=["residual_stress", "shell_layer_stress_cases", "shell_section_z"],
                    scalar_options=["components", "mises"],
                    symbol=r"\boldsymbol{\rho}^{S}",
                    formula=r"\boldsymbol{\rho}^{S}_{k}=\boldsymbol{\rho}^{S}(z_k)",
                    description="Layer Cauchy residual stress on the selected shell surface",
                )
            )
            fields.append(
                field_descriptor(
                    f"shakedown_S_tot_{surface}",
                    label=f"Shakedown S total ({label_suffix})",
                    family="stress",
                    location="gauss",
                    tensor_kind="stress_voigt",
                    n_components=3,
                    frames="frames",
                    frame_axis="vertex",
                    default_selected=surface == "SPOS",
                    source_keys=[
                        "residual_stress",
                        "shell_layer_stress_cases",
                        "shell_section_z",
                        "vertex_load_matrix",
                        "objective_value",
                    ],
                    scalar_options=["components", "mises"],
                    symbol=r"\boldsymbol{\sigma}^{S}_{tot}",
                    formula=r"\boldsymbol{\sigma}^{S}_{v,k}=\boldsymbol{\rho}^{S}_{k}+\alpha\boldsymbol{\sigma}^{E,S}_{v,k}",
                    description="Layer Cauchy total stress reconstructed at each load-domain vertex",
                )
            )
            fields.append(
                field_descriptor(
                    f"shakedown_CInEQ_{surface}",
                    label=f"Shakedown CInEQ ({label_suffix})",
                    family="other",
                    location="gauss",
                    tensor_kind="scalar",
                    n_components=1,
                    frames="frames",
                    frame_axis="vertex",
                    default_selected=True,
                    source_key="yield_ratio",
                    scalar_options=["scalar"],
                    symbol=r"C_{\mathrm{ineq}}",
                    formula=r"C_{\mathrm{ineq},v,k}=\max(\sigma_{vm}(\boldsymbol{\sigma}^{S}_{v,k})/\sigma_y-1,0)",
                    description="Yield inequality violation on the selected shell surface",
                )
            )
    if (is_ilyushin and _has_any(payload, ("residual_stress",))) or (
        is_layer
        and (
            _has_any(payload, ("residual_generalized_stress",))
            or _has_any(payload, ("residual_stress", "shell_layer_stress_cases", "shell_elem_thickness"))
        )
    ):
        for suffix, label, default_selected in (
            ("SF", "Shell generalized membrane force", True),
            ("SM", "Shell generalized bending moment", False),
        ):
            source_keys_res = (
                ["residual_stress"]
                if is_ilyushin
                else ["residual_generalized_stress", "residual_stress", "shell_layer_stress_cases", "shell_section_z", "shell_elem_thickness"]
            )
            source_keys_tot = (
                ["residual_stress", "gauss_stress_cases", "vertex_load_matrix", "objective_value"]
                if is_ilyushin
                else [
                    "residual_generalized_stress",
                    "residual_stress",
                    "shell_generalized_stress_cases",
                    "shell_layer_stress_cases",
                    "shell_section_z",
                    "shell_elem_thickness",
                    "vertex_load_matrix",
                    "objective_value",
                ]
            )
            route_name = "Ilyushin" if is_ilyushin else "layer-integrated"
            fields.append(
                field_descriptor(
                    f"shakedown_{suffix}_res",
                    label=f"Shakedown {suffix} residual",
                    family="stress",
                    location="gauss",
                    tensor_kind="stress_voigt",
                    n_components=3,
                    frames="frames",
                    frame_axis="load_point",
                    default_selected=default_selected,
                    source_keys=source_keys_res,
                    scalar_options=["components", "mises"],
                    symbol=rf"\mathbf{{{suffix}}}_\rho",
                    formula=rf"\mathbf{{{suffix}}}_\rho",
                    description=f"{route_name} generalized {label.lower()} residual",
                )
            )
            fields.append(
                field_descriptor(
                    f"shakedown_{suffix}_tot",
                    label=f"Shakedown {suffix} total",
                    family="stress",
                    location="gauss",
                    tensor_kind="stress_voigt",
                    n_components=3,
                    frames="frames",
                    frame_axis="vertex",
                    default_selected=default_selected,
                    source_keys=source_keys_tot,
                    scalar_options=["components", "mises"],
                    symbol=rf"\mathbf{{{suffix}}}_{{tot}}",
                    formula=rf"\mathbf{{{suffix}}}_v=\mathbf{{{suffix}}}_\rho+\alpha\mathbf{{{suffix}}}^E_v",
                    description=f"{route_name} total generalized {label.lower()} at each load-domain vertex",
                )
            )
    if is_ilyushin and _has_any(payload, ("yield_ratio",)):
        fields.append(
            field_descriptor(
                "shakedown_Phi",
                label="Shakedown Phi (Ilyushin)",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_key="yield_ratio",
                scalar_options=["scalar"],
                symbol=r"\varphi_I",
                formula=r"\varphi_I=\varphi_N+\varphi_M+2c|\varphi_{NM}|",
                description="Ilyushin yield function value; values near 1 indicate active inequality constraints",
            )
        )
        fields.append(
            field_descriptor(
                "shakedown_CInEQ",
                label="Shakedown CInEQ (Ilyushin)",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_key="yield_ratio",
                scalar_options=["scalar"],
                symbol=r"C_{\mathrm{ineq}}",
                formula=r"C_{\mathrm{ineq}}=\max(\varphi_I-1,0)",
                description="Ilyushin yield inequality violation",
            )
        )
    if _has_any(payload, ("yield_ratio",)):
        fields.append(
            field_descriptor(
                "shakedown_inequality_violation",
                label="Shakedown inequality violation",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="vertex",
                default_selected=True,
                source_key="yield_ratio",
                scalar_options=["scalar"],
                symbol=r"C_{\mathrm{ineq}}",
                formula=r"C_{\mathrm{ineq}}=\max(\sigma_{\mathrm{eq}}/\sigma_y-1,0)",
                description="Yield inequality violation; zero means the Gauss point satisfies the yield constraint",
            )
        )
    if _has_any(payload, ("dual_variables_ineq",)):
        fields.append(
            field_descriptor(
                "shakedown_inequality_multiplier",
                label="Yield inequality Lagrange multiplier",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="vertex",
                default_selected=False,
                source_key="dual_variables_ineq",
                scalar_options=["scalar"],
                symbol=r"\lambda_y",
                formula=r"\lambda_y\ge 0,\quad \lambda_y\,g_y(\boldsymbol{\sigma}^{tot})=0",
                description="Dual multiplier of the yield inequality; larger values indicate active plastic-yield constraints",
            )
        )
    if _has_any(payload, ("max_effective_excess_per_gp",)):
        fields.append(
            field_descriptor(
                "max_effective_excess_per_gp",
                label="Max effective excess stress",
                family="other",
                location="gauss",
                tensor_kind="scalar",
                n_components=1,
                frames="frames",
                frame_axis="load_point",
                default_selected=False,
                source_key="max_effective_excess_per_gp",
                scalar_options=["scalar"],
                symbol=r"\sigma^{cs}_{p,vm,\max}",
                formula=r"\sigma^{cs}_{p,vm,\max}=\max_{t\in[0,T]}\sigma^{cs}_{p,vm}(t)",
                description="Maximum von Mises effective excess stress over the RSDM-S cycle per Gauss point",
            )
        )
    if (
        _has_any(payload, ("free_dofs",))
        and (
            _has_any(payload, ("equilibrium_residual",))
            or (_has_any(payload, ("C_sparse",)) and _has_any(payload, ("residual_stress",)))
        )
    ):
        fields.append(
            field_descriptor(
                "shakedown_equality_violation",
                label="Residual self-equilibrium violation",
                family="other",
                location="node",
                tensor_kind="vector",
                n_components=3,
                frames="single",
                default_selected=True,
                source_keys=["equilibrium_residual", "C_sparse", "residual_stress", "free_dofs"],
                scalar_options=["components", "magnitude"],
                symbol=r"C_{\mathrm{eq}}",
                formula=r"C_{\mathrm{eq}}=\mathbf{C}\boldsymbol{\rho}",
                description="Residual-stress self-equilibrium residual on free DOFs",
            )
        )

    if _has(payload, "sigma_E"):
        fields.append(
            field_descriptor(
                "sigma_E",
                label="Elastic stress for shakedown input",
                family="stress",
                location="gauss",
                tensor_kind="stress_voigt",
                n_components=n_comp,
                default_selected=True,
                scalar_options=["components", "mises"],
            )
        )

    return build_viz_manifest(
        producer_module="jaxmech.modules.shakedown",
        analysis_type="shakedown",
        family="shell" if is_shell else "solid",
        fields=fields,
        frame_axes={"vertex": {"label": "Vertex"}, "load_point": {"label": "Load point"}},
        metadata={
            "family": "shell" if is_shell else "solid",
            "shell_yield": shell_yield if is_shell else "",
        },
    )


def build_viz_manifest_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_shakedown_viz_manifest(payload)


def build_viz_manifest_from_mat(mat_path: str | Path) -> dict[str, Any]:
    import scipy.io as sio

    raw = sio.loadmat(str(Path(mat_path)), squeeze_me=False, struct_as_record=True)
    payload = {key: value for key, value in raw.items() if not key.startswith("__")}
    return build_viz_manifest_from_payload(payload)
