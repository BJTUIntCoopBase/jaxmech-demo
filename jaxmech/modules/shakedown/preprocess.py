"""Native MAT-first preprocess for shakedown analyses."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix

from jaxmech.model.result import AnalysisResult

try:
    from jaxmech.io.abaqus.shell_inp import load_shell_model_mat
    _HAS_SHELL_INP = True
except ImportError:
    _HAS_SHELL_INP = False

try:
    from jaxmech.modules.shakedown.shell_operators import build_shell_c_sparse_from_result
    _HAS_SHELL_OPS = True
except ImportError:
    _HAS_SHELL_OPS = False


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
    if arr.size == 0:
        return ""
    return str(arr[0])


def _scalar_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _object_vector(items: list[str]) -> np.ndarray:
    arr = np.empty((len(items),), dtype=object)
    for i, item in enumerate(items):
        arr[i] = str(item)
    return arr


def _as_csr_matrix(value, *, n_rows: int | None = None, n_cols: int | None = None):
    """Recover a SciPy CSR matrix from native or legacy MAT payloads."""
    if hasattr(value, "tocsr"):
        return value.tocsr()

    arr = np.asarray(value)
    if arr.dtype == object and arr.shape == (1, 1):
        obj = arr[0, 0]
        if hasattr(obj, "tocsr"):
            return obj.tocsr()
        if hasattr(obj, "_fieldnames"):
            fieldnames = set(obj._fieldnames)
            if {"indices", "indptr", "data"} <= fieldnames:
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
    """Return True when two sparse matrices match up to floating tolerance."""
    a_csr = a.tocsr()
    b_csr = b.tocsr()
    if a_csr.shape != b_csr.shape:
        return False
    if not np.array_equal(a_csr.indptr, b_csr.indptr):
        return False
    if not np.array_equal(a_csr.indices, b_csr.indices):
        return False
    return np.allclose(a_csr.data, b_csr.data, atol=atol, rtol=rtol)


def _mat_to_native_solid_case(raw: dict) -> dict:
    metadata = _metadata_to_dict(raw.get("metadata"))

    if "C_sparse" in raw and "gauss_stress" in raw and "gauss_coords" in raw and "gauss_vols" in raw and "free_dofs" in raw:
        sigma_e = np.asarray(raw["gauss_stress"], dtype=np.float64)
        if sigma_e.ndim == 3:
            sigma_e = sigma_e.reshape(-1, sigma_e.shape[-1])

        n_str = int(sigma_e.shape[-1])
        n_gauss = int(sigma_e.shape[0])
        n_gauss_per_ele = -1
        n_elem = 1
        if "n_gauss_per_elem" in raw:
            n_gp_arr = np.asarray(raw["n_gauss_per_elem"], dtype=np.int32).reshape(-1)
            n_elem = int(n_gp_arr.size)
            unique = np.unique(n_gp_arr)
            if unique.size == 1:
                n_gauss_per_ele = int(unique[0])
        if "solid_elem_types" in raw:
            elem_types = np.asarray(raw["solid_elem_types"], dtype=object).reshape(-1)
        else:
            elem_types = _object_vector(["UNKNOWN"] * max(1, n_elem))

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
            "validated_with_ODB": _scalar_int(raw.get("validated_with_ODB", [[0]])),
            "validated_ODBName": _scalar_string(raw.get("validated_ODBName", np.asarray([""]))),
            "error_with_odb": raw.get("Error_with_ODB"),
        }

    if "C_sparse_jax" in raw and "free_dofs_jax" in raw:
        free_dofs = np.asarray(raw["free_dofs_jax"], dtype=np.int64).reshape(-1)
        if "EleGaussStressSet_jax" in raw:
            sigma_ele = np.asarray(raw["EleGaussStressSet_jax"], dtype=np.float64)
            sigma_e = sigma_ele.reshape(-1, sigma_ele.shape[-1])
        else:
            sigma_parts = [
                np.asarray(obj, dtype=np.float64).reshape(-1, np.asarray(obj, dtype=np.float64).shape[-1])
                for obj in raw["EleGaussStressSetByType_jax"].reshape(-1)
            ]
            sigma_e = np.concatenate(sigma_parts, axis=0)

        if "gauss_coords_jax" in raw:
            gauss_coords = np.asarray(raw["gauss_coords_jax"], dtype=np.float64)
        else:
            gauss_coords = np.concatenate(
                [
                    np.asarray(obj, dtype=np.float64).reshape(-1, np.asarray(obj, dtype=np.float64).shape[-1])
                    for obj in raw["gauss_coords_by_type_jax"].reshape(-1)
                ],
                axis=0,
            )
        if "gauss_vols_jax" in raw:
            gauss_vols = np.asarray(raw["gauss_vols_jax"], dtype=np.float64).reshape(-1)
        else:
            gauss_vols = np.concatenate(
                [np.asarray(obj, dtype=np.float64).reshape(-1) for obj in raw["gauss_vols_by_type_jax"].reshape(-1)],
                axis=0,
            )

        return {
            "sigma_E": sigma_e,
            "C_sparse": _as_csr_matrix(
                raw["C_sparse_jax"],
                n_rows=int(free_dofs.size),
                n_cols=int(sigma_e.shape[0] * sigma_e.shape[1]),
            ),
            "gauss_coords": gauss_coords,
            "gauss_vols": gauss_vols,
            "free_dofs": free_dofs,
            "n_gauss": int(gauss_coords.shape[0]),
            "n_gauss_per_ele": int(np.asarray(raw.get("n_gauss_per_elem_jax", [[-1]])).reshape(-1)[0]),
            "n_str": int(np.asarray(raw["n_str_jax"]).reshape(-1)[0]),
            "elem_types": _object_vector(["UNKNOWN"]),
            "use_b_ext": int(np.asarray(raw.get("use_b_ext_jax", [[0]])).reshape(-1)[0]),
            "metadata": metadata,
            "validated_with_ODB": _scalar_int(raw.get("validated_with_ODB", [[0]])),
            "validated_ODBName": _scalar_string(raw.get("validated_ODBName", np.asarray([""]))),
            "error_with_odb": raw.get("Error_with_ODB"),
        }

    raise KeyError(
        "Unsupported solid MAT payload for shakedown preprocess. "
        "Expected either native MAT-first fields or legacy *_jax shakedown fields."
    )


def _unwrap_mat_scalar(value):
    out = value
    while isinstance(out, np.ndarray) and out.dtype == object and out.size == 1:
        out = out.reshape(-1)[0]
    return out


def _mat_struct_to_payload(value):
    out = _unwrap_mat_scalar(value)
    if isinstance(out, dict):
        return out
    if hasattr(out, "__dict__"):
        return {k: v for k, v in out.__dict__.items() if not k.startswith("_")}
    return out


def _shell_info_from_raw(raw: dict) -> dict:
    inp = raw.get("InpData", {})
    if not isinstance(inp, dict):
        inp = _mat_struct_to_payload(inp)
    ele_type = str(np.asarray(inp.get("ele_type", np.asarray(["S4"]))).reshape(-1)[0])
    thickness = float(np.asarray(inp.get("thickness", np.asarray([0.0])), dtype=np.float64).reshape(-1)[0])
    material_name = str(np.asarray(inp.get("material_name", np.asarray([""]))).reshape(-1)[0])
    return {
        "ele_type": ele_type,
        "thickness": thickness,
        "material_name": material_name,
    }


def _expand_elem_values_to_gauss(values: np.ndarray, n_gauss_per_elem: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    counts = np.asarray(n_gauss_per_elem, dtype=np.int32).reshape(-1)
    if values.size != counts.size:
        raise ValueError(
            f"Element-wise value count {values.size} does not match n_gauss_per_elem length {counts.size}."
        )
    return np.repeat(values, counts)


def _mat_to_native_shell_case(raw: dict, mat_path: Path) -> dict:
    result = AnalysisResult.from_mat(str(mat_path))
    if result.operators.C_gen is None:
        raise KeyError(f"Shell shakedown requires C_gen in MAT: {mat_path}")
    if result.shell is None:
        raise KeyError(f"Shell shakedown requires shell recovery fields in MAT: {mat_path}")
    if result.shell.layer_stress is None:
        raise KeyError(f"Shell shakedown requires shell_layer_stress in MAT: {mat_path}")
    if result.shell.elem_thickness is None:
        raise KeyError(f"Shell shakedown requires shell_elem_thickness in MAT: {mat_path}")
    if result.shell.section_z is None:
        raise KeyError(f"Shell shakedown requires shell_section_z in MAT: {mat_path}")
    if result.n_gauss_per_elem is None:
        raise KeyError(f"Shell shakedown requires n_gauss_per_elem in MAT: {mat_path}")
    if result.gauss_stress is None:
        raise KeyError(f"Shell shakedown requires gauss_stress in MAT: {mat_path}")
    if result.gauss_coords is None:
        raise KeyError(f"Shell shakedown requires gauss_coords in MAT: {mat_path}")

    raw_meta = _metadata_to_dict(raw.get("metadata"))
    shell_info = _shell_info_from_raw(raw)
    c_gen = result.operators.C_gen.tocsr()
    layer_stress = np.asarray(result.shell.layer_stress, dtype=np.float64)
    if layer_stress.ndim != 4 or layer_stress.shape[-1] != 3:
        raise ValueError(
            f"shell_layer_stress must have shape (n_elem, n_gp, n_section_points, 3), got {layer_stress.shape}."
        )
    n_elem, n_gp, n_sp, _ = layer_stress.shape
    counts = np.asarray(result.n_gauss_per_elem, dtype=np.int32).reshape(-1)
    if counts.size != n_elem:
        raise ValueError(
            f"n_gauss_per_elem length {counts.size} does not match shell elements {n_elem} in {mat_path}."
        )
    unique = np.unique(counts)
    if unique.size != 1 or int(unique[0]) != n_gp:
        raise NotImplementedError(
            f"Mixed shell gauss layout is not supported in native shakedown: {unique.tolist()}."
        )

    sigma_e_gen = np.asarray(result.gauss_stress, dtype=np.float64).reshape(n_elem * n_gp, 6)
    sigma_e_layers = layer_stress.reshape(n_elem * n_gp, n_sp, 3)
    elem_thickness = np.asarray(result.shell.elem_thickness, dtype=np.float64).reshape(-1)
    if elem_thickness.size != n_elem:
        raise ValueError(
            f"shell_elem_thickness length {elem_thickness.size} does not match shell elements {n_elem} in {mat_path}."
        )
    gauss_thickness = _expand_elem_values_to_gauss(elem_thickness, counts)

    return {
        "sigma_E_gen": sigma_e_gen,
        "sigma_E_layers": sigma_e_layers,
        "C_gen": c_gen,
        "C_sparse": build_shell_c_sparse_from_result(result),
        "free_dofs": np.asarray(result.operators.free_dofs, dtype=np.int64).reshape(-1)
        if result.operators.free_dofs is not None
        else np.empty((0,), dtype=np.int64),
        "gauss_coords": np.asarray(result.gauss_coords, dtype=np.float64).reshape(n_elem * n_gp, -1),
        "n_gauss": int(n_elem * n_gp),
        "n_gauss_per_elem": counts,
        "n_layer": int(n_sp),
        "section_z": np.asarray(result.shell.section_z, dtype=np.float64).reshape(-1),
        "elem_thickness": elem_thickness,
        "gauss_thickness": gauss_thickness,
        "elem_types": _object_vector([str(raw_meta.get("ele_type", shell_info["ele_type"]))] * n_elem),
        "ele_type": str(raw_meta.get("ele_type", shell_info["ele_type"])),
        "material_name": shell_info["material_name"],
        "metadata": raw_meta,
        "validated_with_ODB": _scalar_int(raw.get("validated_with_ODB", [[0]])),
        "validated_ODBName": _scalar_string(raw.get("validated_ODBName", np.asarray([""]))),
        "error_with_odb": raw.get("Error_with_ODB"),
    }


def _build_ele_yield(case: dict, yield_values) -> np.ndarray:
    if isinstance(yield_values, (int, float)):
        return np.full((case["n_gauss"],), float(yield_values), dtype=np.float64)
    if isinstance(yield_values, dict) and len(yield_values) == 1:
        return np.full((case["n_gauss"],), float(next(iter(yield_values.values()))), dtype=np.float64)
    raise ValueError(
        "Native solid shakedown preprocess currently expects a scalar yield_values "
        "or a single-material dict. Multi-material mapping will be added in the "
        "native builder stage."
    )


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
        "metadata": {
            "source": "jax_mat",
            "use_b_ext": int(base["use_b_ext"]),
        },
        "elem_types": np.asarray(base["elem_types"], dtype=object).reshape(-1),
    }
    if len(sigma_cases) > 1:
        fem_input["sigma_E_cases"] = np.stack(sigma_cases[1:], axis=2)
    return fem_input


def build_shell_fem_input(
    case_mats: list[dict],
    case_paths: list[Path],
    yield_values,
    shell_yield: str,
    ilyushin_c: float = 0.5,
) -> dict:
    native_cases = [_mat_to_native_shell_case(raw, mat_path) for raw, mat_path in zip(case_mats, case_paths)]
    base = native_cases[0]

    for other in native_cases[1:]:
        if base["ele_type"] != other["ele_type"]:
            raise ValueError("ele_type mismatch between shell MAT cases.")
        if not np.array_equal(base["elem_types"], other["elem_types"]):
            raise ValueError("elem_types mismatch between shell MAT cases.")
        if not _sparse_allclose(base["C_gen"], other["C_gen"]):
            raise ValueError("C_gen mismatch between shell MAT cases.")
        if not _sparse_allclose(base["C_sparse"], other["C_sparse"]):
            raise ValueError("C_sparse mismatch between shell MAT cases.")
        if not np.array_equal(base["free_dofs"], other["free_dofs"]):
            raise ValueError("free_dofs mismatch between shell MAT cases.")
        if not np.allclose(base["gauss_coords"], other["gauss_coords"], atol=1e-12, rtol=1e-12):
            raise ValueError("gauss_coords mismatch between shell MAT cases.")
        if not np.array_equal(base["n_gauss_per_elem"], other["n_gauss_per_elem"]):
            raise ValueError("n_gauss_per_elem mismatch between shell MAT cases.")
        if not np.allclose(base["section_z"], other["section_z"], atol=1e-12, rtol=1e-12):
            raise ValueError("shell_section_z mismatch between shell MAT cases.")
        if not np.allclose(base["elem_thickness"], other["elem_thickness"], atol=1e-12, rtol=1e-12):
            raise ValueError("shell_elem_thickness mismatch between shell MAT cases.")

    sigma_y = float(_build_ele_yield({"n_gauss": 1}, yield_values)[0])
    fem_input = {
        "sigma_y": sigma_y,
        "free_dofs": np.asarray(base["free_dofs"], dtype=np.int64).reshape(-1),
        "gauss_coords": np.asarray(base["gauss_coords"], dtype=np.float64),
        "n_gauss": int(base["n_gauss"]),
        "n_gauss_per_elem": np.asarray(base["n_gauss_per_elem"], dtype=np.int32),
        "ele_type": base["ele_type"],
        "elem_types": np.asarray(base["elem_types"], dtype=object).reshape(-1),
        "material_name": base["material_name"],
        "section_z": np.asarray(base["section_z"], dtype=np.float64),
        "elem_thickness": np.asarray(base["elem_thickness"], dtype=np.float64),
        "gauss_thickness": np.asarray(base["gauss_thickness"], dtype=np.float64),
        "metadata": {"source": "jax_mat"},
        "shell_yield": str(shell_yield),
        "ilyushin_c": float(ilyushin_c),
        "shell_model": load_shell_model_mat(case_paths[0]),
    }
    if str(shell_yield).lower() == "ilyushin":
        fem_input["C_gen"] = base["C_gen"].tocsr()
        fem_input["sigma_E_gen"] = np.asarray(base["sigma_E_gen"], dtype=np.float64)
        if len(native_cases) > 1:
            fem_input["sigma_E_gen_cases"] = np.stack(
                [np.asarray(case["sigma_E_gen"], dtype=np.float64) for case in native_cases[1:]],
                axis=2,
            )
    else:
        fem_input["C_sparse"] = base["C_sparse"].tocsr()
        fem_input["sigma_E_layers"] = np.asarray(base["sigma_E_layers"], dtype=np.float64)
        if len(native_cases) > 1:
            fem_input["sigma_E_layers_cases"] = np.stack(
                [np.asarray(case["sigma_E_layers"], dtype=np.float64) for case in native_cases[1:]],
                axis=3,
            )
    return fem_input


def resolve_case_io(config: dict) -> list[tuple[str, Path, Path]]:
    case_io: list[tuple[str, Path, Path]] = []
    for mat_file in config["mat_files"]:
        mat_path = Path(mat_file)
        case_name = mat_path.stem
        case_io.append((case_name, Path(), mat_path))
    return case_io


def load_case_mats(case_io: list[tuple[str, Path, Path]]) -> list[dict]:
    mats: list[dict] = []
    for case_name, _, mat_path in case_io:
        if not mat_path.is_file():
            raise FileNotFoundError(
                f"JAX MAT not found for {case_name}: {mat_path}. "
                "Run the elastic build step first so shakedown starts from MAT."
            )
        mats.append(_load_case_mat(mat_path))
    return mats


def case_mat_paths(case_io: list[tuple[str, Path, Path]]) -> list[Path]:
    return [mat_path for _, _, mat_path in case_io]


def _load_factor_rows(config: dict, n_loads: int) -> list[np.ndarray]:
    rows = np.asarray(config["load_factor_set"], dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"load_factor_set must be a 2-D array, got shape {rows.shape}.")
    if rows.shape[1] < n_loads:
        # Pad with zeros on the right to reach n_loads columns
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
            return [
                {
                    "label": f"theta_{theta:g}",
                    "theta_deg": float(theta),
                    "phi_deg": None,
                    "load_scale": None,
                }
                for theta in config["theta_deg"]
            ]
        if n_loads == 3:
            return [
                {
                    "label": f"theta_{theta:g}_phi_{phi:g}",
                    "theta_deg": float(theta),
                    "phi_deg": float(phi),
                    "load_scale": None,
                }
                for theta, phi in itertools.product(config["theta_deg"], config["phi_deg"])
            ]
        raise ValueError(f"Unsupported number of load cases: {n_loads}")

    rows = _load_factor_rows(config, n_loads)
    return [
        {
            "label": "lf_" + "_".join(f"{x:g}" for x in row),
            "theta_deg": None,
            "phi_deg": None,
            "load_scale": row,
        }
        for row in rows
    ]
