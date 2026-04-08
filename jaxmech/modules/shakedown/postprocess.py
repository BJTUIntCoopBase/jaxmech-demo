"""Postprocessing helpers for shakedown summary MAT files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from jaxmech.modules.shakedown.result import ShakedownResult


def _extract_alpha_values(raw: dict) -> list[float]:
    results_set = raw.get("ResultsSet")
    if results_set is None:
        return []

    alpha_values: list[float] = []
    for i in range(results_set.shape[0]):
        entry = results_set[i, 0]
        alpha_val = float(np.asarray(entry.objective_value).reshape(-1)[0])
        alpha_values.append(alpha_val)
    return alpha_values


def load_shakedown_summary(
    mat_path: str | Path,
    *,
    solver: str,
    config_path: str | Path | None = None,
) -> ShakedownResult:
    """Build ``ShakedownResult`` from one summary MAT file."""
    mat_path = Path(mat_path)
    raw = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    alpha_values = _extract_alpha_values(raw)
    best_alpha = max(alpha_values) if alpha_values else np.nan

    return ShakedownResult(
        alpha=best_alpha,
        status="optimal" if alpha_values else "unknown",
        solver=str(solver),
        summary_mat_path=str(mat_path),
        metadata={
            "n_parameter_sets": len(alpha_values),
            "all_alphas": alpha_values,
            "config_path": str(config_path) if config_path is not None else None,
        },
    )


def _dicts_to_struct_array(items: list[dict]) -> np.ndarray:
    if not items:
        return np.empty((0, 1), dtype=object)
    field_names = list(items[0].keys())
    dtype = [(name, object) for name in field_names]
    arr = np.empty((len(items), 1), dtype=dtype)
    for i, item in enumerate(items):
        for name in field_names:
            arr[name][i, 0] = item[name]
    return arr


def save_shakedown_summary(
    out_path: str | Path,
    *,
    elastic_input: dict,
    results: list[dict],
    config_info: dict | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save one shakedown summary MAT with minimal native structure."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ElasticInputSet": elastic_input,
        "ResultsSet": _dicts_to_struct_array(results),
    }
    if config_info:
        payload["ConfigInfo"] = config_info
    if metadata:
        for key, value in metadata.items():
            payload[f"meta_{key}"] = np.asarray([value]) if isinstance(value, str) else np.asarray(value)
    sio.savemat(str(out_path), payload, do_compression=True)
    return out_path
