"""MAT-first preprocess for solid demo shakedown analyses."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix


def _load_case_mat(mat_path: Path) -> dict:
    return sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)


def _metadata_to_dict(raw_meta) -> dict[str, object]:
    if raw_meta is None:
        return {}
    if isinstance(raw_meta, dict):
        return raw_meta
    if hasattr(raw_meta, "_fieldnames"):
        return {name: getattr(raw_meta, name) for name in raw_meta._fieldnames}
    if isinstance(raw_meta, np.ndarray) and getattr(raw_meta.dtype, "names", None):
        rec = raw_meta.reshape(-1)[0]
        return {name: rec[name] for name in raw_meta.dtype.names}
    return {}


def _scalar_string(value) -> str:
    arr = np.asarray(value).reshape(-1)
    return "" if arr.size == 0 else str(arr[0])


def _object_vector(items: list[str]) -> np.ndarray:
    arr = np.empty((len(items),), dtype=object)
    for i, item in enumerate(items):
        arr[i] = str(item)
    return arr


def _as_csr_matrix(value, *, n_rows: int | None = None, n_cols: int | None = None):
    if hasattr(value, "tocsr"):
        return value.tocsr()
    arr = np.asarray(value)
    if arr.dtype == object and arr.shape == (1, 1):
        obj = arr[0, 0]
        if hasattr(obj, "tocsr"):
            return obj.tocsr()
        if hasattr(obj, "_fieldnames") and {"indices", "indptr", "data"} <= set(obj._fieldnames):
            indices = np.asarray(getattr(obj, "indices"), dtype=np.int32).reshape(-1)
            indptr = np.asarray(getattr(obj, "indptr"), dtype=np.int32).reshape(-1)
            data = np.asarray(getattr(obj, "data"), dtype=np.float64).reshape(-1)
            if n_cols is None:
                n_cols = int(indptr.size - 1)
            if n_rows is None:
                n_rows = int(indices.max() + 1) if indices.size else 0
            return csc_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.float64).tocsr()
    raise TypeError(f"Unsupported sparse matrix payload type: {type(value)!r}")


def _sparse_allclose(a, b, *, atol: float = 1e-12, rtol: float = 1e-12) -> bool:
    a_csr = a.tocsr()
    b_csr = b.tocsr()
    return (
        a_csr.shape == b_csr.shape
        and np.array_equal(a_csr.indptr, b_csr.indptr)
        and np.array_equal(a_csr.indices, b_csr.indices)
        and np.allclose(a_csr.data, b_csr.data, atol=atol, rtol=rtol)
    )


def _mat_to_native_solid_case(raw: dict) -> dict:
    metadata = _metadata_to_dict(raw.get("metadata"))
    family = str(np.asarray(metadata.get("family", "solid")).reshape(-1)[0]).strip().lower() if metadata else "solid"
    if family != "solid":
        raise ValueError(f"Only solid MAT files are supported by the demo shakedown workflow, got {family!r}.")

    if {"C_sparse", "gauss_stress", "gauss_coords", "gauss_vols", "free_dofs"} <= set(raw):
        sigma_e = np.asarray(raw["gauss_stress"], dtype=np.float64)
        if sigma_e.ndim == 3:
            sigma_e = sigma_e.reshape(-1, sigma_e.shape[-1])
        n_gauss = int(sigma_e.shape[0])
        n_str = int(sigma_e.shape[-1])
        n_gauss_per_ele = -1
        n_elem = 1
        if "n_gauss_per_elem" in raw:
            counts = np.asarray(raw["n_gauss_per_elem"], dtype=np.int32).reshape(-1)
            n_elem = int(counts.size)
            unique = np.unique(counts)
            if unique.size == 1:
                n_gauss_per_ele = int(unique[0])
        elem_types = np.asarray(raw.get("solid_elem_types", _object_vector(["UNKNOWN"] * max(1, n_elem))), dtype=object).reshape(-1)
        return {
            "sigma_E": sigma_e,
            "C_sparse": _as_csr_matrix(raw["C_sparse"]),
            "gauss_coords": np.asarray(raw["gauss_coords"], dtype=np.float64),
            "gauss_vols": np.asarray(raw["gauss_vols"], dtype=np.float64).reshape(-1),
            "free_dofs": np.asarray(raw["free_dofs"], dtype=np.int64).reshape(-1),
            "n_gauss": n_gauss,
            "n_gauss_per_ele": n_gauss_per_ele,
            "n_str": n_str,
            "elem_types": elem_types,
            "use_b_ext": int(ast.literal_eval(_scalar_string(metadata.get("use_b_ext", np.asarray(["0"]))))),
            "metadata": metadata,
        }

    if "C_sparse_jax" in raw and "free_dofs_jax" in raw:
        free_dofs = np.asarray(raw["free_dofs_jax"], dtype=np.int64).reshape(-1)
        if "EleGaussStressSet_jax" in raw:
            sigma_ele = np.asarray(raw["EleGaussStressSet_jax"], dtype=np.float64)
            sigma_e = sigma_ele.reshape(-1, sigma_ele.shape[-1])
        else:
            sigma_e = np.concatenate([
                np.asarray(obj, dtype=np.float64).reshape(-1, np.asarray(obj, dtype=np.float64).shape[-1])
                for obj in raw["EleGaussStressSetByType_jax"].reshape(-1)
            ], axis=0)
        gauss_coords = np.asarray(raw["gauss_coords_jax"], dtype=np.float64)
        gauss_vols = np.asarray(raw["gauss_vols_jax"], dtype=np.float64).reshape(-1)
        return {
            "sigma_E": sigma_e,
            "C_sparse": _as_csr_matrix(raw["C_sparse_jax"], n_rows=int(free_dofs.size), n_cols=int(sigma_e.shape[0] * sigma_e.shape[1])),
            "gauss_coords": gauss_coords,
            "gauss_vols": gauss_vols,
            "free_dofs": free_dofs,
            "n_gauss": int(gauss_coords.shape[0]),
            "n_gauss_per_ele": int(np.asarray(raw.get("n_gauss_per_elem_jax", [[-1]])).reshape(-1)[0]),
            "n_str": int(np.asarray(raw["n_str_jax"]).reshape(-1)[0]),
            "elem_types": _object_vector(["UNKNOWN"]),
            "use_b_ext": int(np.asarray(raw.get("use_b_ext_jax", [[0]])).reshape(-1)[0]),
            "metadata": metadata,
        }

    raise KeyError("Unsupported solid MAT payload for shakedown preprocess.")


def _build_ele_yield(case: dict, yield_values) -> np.ndarray:
    if isinstance(yield_values, (int, float)):
        return np.full((case["n_gauss"],), float(yield_values), dtype=np.float64)
    if isinstance(yield_values, dict) and len(yield_values) == 1:
        return np.full((case["n_gauss"],), float(next(iter(yield_values.values()))), dtype=np.float64)
    raise ValueError("Demo shakedown expects a scalar yield_values or a single-material dictionary.")


def build_solid_fem_input(case_mats: list[dict], yield_values) -> dict:
    native_cases = [_mat_to_native_solid_case(raw) for raw in case_mats]
    base = native_cases[0]
    for other in native_cases[1:]:
        if not _sparse_allclose(base["C_sparse"], other["C_sparse"]):
            raise ValueError("C_sparse mismatch between load cases.")
        if not np.allclose(base["gauss_coords"], other["gauss_coords"], atol=1e-12, rtol=1e-12):
            raise ValueError("gauss_coords mismatch between load cases.")
        if not np.allclose(base["gauss_vols"], other["gauss_vols"], atol=1e-12, rtol=1e-12):
            raise ValueError("gauss_vols mismatch between load cases.")
        if not np.array_equal(base["free_dofs"], other["free_dofs"]):
            raise ValueError("free_dofs mismatch between load cases.")
        if base["n_str"] != other["n_str"]:
            raise ValueError("n_str mismatch between load cases.")

    sigma_cases = [np.asarray(case["sigma_E"], dtype=np.float64) for case in native_cases]
    fem_input = {
        "sigma_E": sigma_cases[0],
        "C_sparse": base["C_sparse"].tocsr(),
        "ele_yield": _build_ele_yield(base, yield_values),
        "free_dofs": np.asarray(base["free_dofs"], dtype=np.int64).reshape(-1),
        "gauss_coords": np.asarray(base["gauss_coords"], dtype=np.float64),
        "gauss_vols": np.asarray(base["gauss_vols"], dtype=np.float64).reshape(-1),
        "n_gauss": int(base["n_gauss"]),
        "n_gauss_per_ele": int(base["n_gauss_per_ele"]),
        "n_str": int(base["n_str"]),
        "metadata": {"source": "jax_mat", "use_b_ext": int(base["use_b_ext"])},
        "elem_types": np.asarray(base["elem_types"], dtype=object).reshape(-1),
    }
    if len(sigma_cases) > 1:
        fem_input["sigma_E_cases"] = np.stack(sigma_cases[1:], axis=2)
    return fem_input


def resolve_case_io(config: dict) -> list[tuple[str, Path, Path]]:
    return [(Path(mat_file).stem, Path(), Path(mat_file)) for mat_file in config["mat_files"]]


def load_case_mats(case_io: list[tuple[str, Path, Path]]) -> list[dict]:
    mats: list[dict] = []
    for case_name, _, mat_path in case_io:
        if not mat_path.is_file():
            raise FileNotFoundError(f"JAX MAT not found for {case_name}: {mat_path}. Run elastic analysis first.")
        mats.append(_load_case_mat(mat_path))
    return mats


def _load_factor_rows(config: dict, n_loads: int) -> list[np.ndarray]:
    rows = np.asarray(config["load_factor_set"], dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"load_factor_set must be 2-D, got {rows.shape}.")
    if rows.shape[1] < n_loads:
        pad = np.zeros((rows.shape[0], n_loads - rows.shape[1]), dtype=np.float64)
        rows = np.concatenate([rows, pad], axis=1)
    return [row[:n_loads].copy() for row in rows]


def build_parameter_sets(config: dict, case_io: list[tuple[str, Path, Path]], case_raw: list[dict]) -> list[dict]:
    del case_io, case_raw
    n_loads = len(config["load_cases"])
    if config["use_angles"]:
        if n_loads == 1:
            return [{"label": "single", "theta_deg": None, "phi_deg": None, "load_scale": None}]
        if n_loads == 2:
            return [{"label": f"theta_{theta:g}", "theta_deg": float(theta), "phi_deg": None, "load_scale": None} for theta in config["theta_deg"]]
        if n_loads == 3:
            return [
                {"label": f"theta_{theta:g}_phi_{phi:g}", "theta_deg": float(theta), "phi_deg": float(phi), "load_scale": None}
                for theta, phi in itertools.product(config["theta_deg"], config["phi_deg"])
            ]
        raise ValueError(f"Unsupported number of load cases: {n_loads}")
    return [{"label": "lf_" + "_".join(f"{x:g}" for x in row), "theta_deg": None, "phi_deg": None, "load_scale": row} for row in _load_factor_rows(config, n_loads)]
