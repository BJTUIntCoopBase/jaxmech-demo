"""Build a PyVista UnstructuredGrid from ``VizData``.

Also provides Gauss-point-to-node averaging so that element-wise
Gauss fields can be displayed as smooth nodal contour plots.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None  # type: ignore[assignment]

from jaxmech.modules.visualization.mat_reader import VizData

# Typical Gauss-point counts per ABAQUS element type.
_GP_PER_ELEM: dict[str, int] = {
    "C3D4": 1, "C3D10": 4, "C3D6": 2,
    "C3D8": 8, "C3D8R": 1, "C3D8I": 8,
    "C3D20": 27, "C3D20R": 8,
    "CPE3": 1, "CPS3": 1,
    "CPE4": 4, "CPE4R": 1, "CPS4": 4, "CPS4R": 1,
    "CPE6": 3, "CPS6": 3,
    "CPE8": 9, "CPS8": 9,
}

# Nodes per element for VTK cell types we handle.
_NODES_PER_VTK: dict[int, int] = {
    5: 3, 9: 4, 10: 4, 12: 8, 13: 6,
    22: 6, 23: 8, 24: 10, 25: 20,
}


def _element_type_key(value: object) -> str:
    """Return a normalized ABAQUS element type name from MAT cell-like values."""
    try:
        arr = np.asarray(value, dtype=object)
        if arr.size:
            value = arr.reshape(-1)[0]
    except Exception:
        pass
    if isinstance(value, bytes):
        text = value.decode(errors="ignore")
    else:
        text = str(value)
    return text.strip().strip("'\"").upper()


def _cell_types_1d(vd: VizData) -> np.ndarray:
    return np.asarray(vd.cell_types, dtype=np.int32).reshape(-1)


def build_grid(vd: VizData) -> "pv.UnstructuredGrid":
    """Create a ``pyvista.UnstructuredGrid`` from *vd*.

    Parameters
    ----------
    vd : VizData
        Loaded visualisation data (must have mesh geometry).

    Returns
    -------
    pyvista.UnstructuredGrid

    Raises
    ------
    RuntimeError
        If PyVista is not installed or *vd* has no mesh.
    """
    if pv is None:
        raise RuntimeError("PyVista is required: pip install pyvista")
    if not vd.has_mesh:
        raise RuntimeError("VizData has no mesh geometry (points/cells missing).")

    points = vd.points
    if points.shape[1] == 2:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])

    grid = pv.UnstructuredGrid(vd.cells, _cell_types_1d(vd), points)
    return grid


# ---------------------------------------------------------------------------
# Gauss -> element-average -> node averaging
# ---------------------------------------------------------------------------

def _elem_offsets(vd: VizData) -> np.ndarray:
    """Return the number of Gauss points per element as an int array."""
    if vd.n_gauss_per_elem is not None:
        return np.asarray(vd.n_gauss_per_elem, dtype=np.int32).reshape(-1)

    if vd.cell_ele_types is not None:
        eletypes = vd.cell_ele_types
        ngpe = np.array(
            [_GP_PER_ELEM.get(_element_type_key(e), 1) for e in eletypes],
            dtype=np.int32,
        )
        return ngpe

    n_cells = int(_cell_types_1d(vd).shape[0])
    return np.ones(n_cells, dtype=np.int32)


def _normalize_gauss_field_layout(field: np.ndarray, vd: VizData) -> np.ndarray:
    """Convert element-gauss layouts to the flat gauss layout expected downstream.

    Visualization MATs can store gauss fields either as:
    - ``(n_elem, n_gp, n_comp)`` / ``(n_elem, n_gp)`` for homogeneous meshes
    - ``(n_gauss_total, n_comp)`` / ``(n_gauss_total,)`` for flat layouts

    The gauss-to-node path expects the latter, so normalize before averaging.
    """
    arr = np.asarray(field, dtype=np.float64)
    ngpe = _elem_offsets(vd)
    n_elem = int(ngpe.shape[0])

    if arr.ndim == 3 and arr.shape[0] == n_elem:
        return arr.reshape(-1, arr.shape[-1])

    if arr.ndim == 2 and arr.shape[0] == n_elem and np.all(ngpe == arr.shape[1]):
        return arr.reshape(-1)

    return arr


def gauss_to_element_avg(field: np.ndarray, vd: VizData) -> np.ndarray:
    """Average Gauss-point values to one value per element.

    Parameters
    ----------
    field : ndarray
        Shape ``(n_gauss_total, n_comp)`` or ``(n_gauss_total,)``.
    vd : VizData
        Must have mesh info.

    Returns
    -------
    ndarray
        Shape ``(n_elements, n_comp)`` or ``(n_elements,)``.
    """
    field = _normalize_gauss_field_layout(field, vd)

    squeeze = False
    if field.ndim == 1:
        field = field[:, np.newaxis]
        squeeze = True

    ngpe = _elem_offsets(vd)
    n_elem = ngpe.shape[0]
    n_comp = field.shape[1]
    out = np.empty((n_elem, n_comp), dtype=np.float64)

    idx = 0
    for i in range(n_elem):
        g = int(ngpe[i])
        out[i] = field[idx:idx + g].mean(axis=0)
        idx += g

    return out.squeeze(axis=1) if squeeze else out


def gauss_to_nodal(field: np.ndarray, vd: VizData) -> np.ndarray:
    """Average Gauss values to nodes via element averaging + scatter.

    1. Average Gauss points within each element.
    2. Scatter the element average to each node of the element.
    3. Average contributions from all elements sharing a node.

    Parameters
    ----------
    field : ndarray
        Gauss-point field, ``(n_gauss_total, n_comp)`` or flat.
    vd : VizData
        Must have mesh and cell connectivity.

    Returns
    -------
    ndarray
        Nodal field, ``(n_nodes, n_comp)`` or ``(n_nodes,)``.
    """
    field = _normalize_gauss_field_layout(field, vd)

    squeeze = False
    if field.ndim == 1:
        field = field[:, np.newaxis]
        squeeze = True

    elem_avg = gauss_to_element_avg(field, vd)
    if elem_avg.ndim == 1:
        elem_avg = elem_avg[:, np.newaxis]
    n_nodes = int(vd.points.shape[0])
    n_comp = elem_avg.shape[1]

    accum = np.zeros((n_nodes, n_comp), dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.int32)

    cells = vd.cells
    ctypes = _cell_types_1d(vd)
    n_elem = ctypes.shape[0]

    pos = 0
    for i in range(n_elem):
        npe = int(cells[pos])
        node_ids = cells[pos + 1: pos + 1 + npe].astype(np.intp)
        accum[node_ids] += elem_avg[i]
        count[node_ids] += 1
        pos += 1 + npe

    mask = count > 0
    accum[mask] /= count[mask, np.newaxis]
    return accum.squeeze(axis=1) if squeeze else accum


def element_to_nodal(field: np.ndarray, vd: VizData) -> np.ndarray:
    """Scatter element-wise values to connected nodes by averaging."""
    arr = np.asarray(field, dtype=np.float64)
    squeeze = False
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
        squeeze = True

    n_nodes = int(vd.points.shape[0])
    n_comp = int(arr.shape[1])
    accum = np.zeros((n_nodes, n_comp), dtype=np.float64)
    count = np.zeros(n_nodes, dtype=np.int32)

    cells = vd.cells
    ctypes = _cell_types_1d(vd)
    n_elem = int(ctypes.shape[0])
    if arr.shape[0] < n_elem:
        arr = np.pad(arr, ((0, n_elem - arr.shape[0]), (0, 0)))

    pos = 0
    for i in range(n_elem):
        npe = int(cells[pos])
        node_ids = cells[pos + 1: pos + 1 + npe].astype(np.intp)
        accum[node_ids] += arr[i]
        count[node_ids] += 1
        pos += 1 + npe

    mask = count > 0
    accum[mask] /= count[mask, np.newaxis]
    return accum.squeeze(axis=1) if squeeze else accum


def scalarize_field(
    field: np.ndarray,
    vd: VizData,
    location: str = "gauss",
    *,
    component: Optional[int] = None,
    field_key: Optional[str] = None,
) -> np.ndarray:
    """Reduce a raw field to the scalar nodal values used for rendering."""
    arr = np.asarray(field, dtype=np.float64)
    if location == "gauss":
        arr = _normalize_gauss_field_layout(arr, vd)
    elif location == "element":
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    n_comp = arr.shape[-1]
    field_kind = _classify_field_kind(field_key)

    if component is not None and 0 <= component < n_comp:
        scalar = arr[..., component]
    elif field_kind == "stress" and n_comp == 6:
        scalar = _von_mises_6(arr)
    elif field_kind == "stress" and n_comp == 3:
        scalar = _von_mises_3(arr)
    elif field_kind == "strain" and n_comp == 6:
        scalar = _equivalent_strain_6(arr)
    elif field_kind == "strain" and n_comp == 3:
        scalar = _equivalent_strain_3(arr)
    elif n_comp == 3:
        scalar = np.linalg.norm(arr, axis=-1)
    elif n_comp > 1:
        scalar = np.linalg.norm(arr, axis=-1)
    else:
        scalar = arr[..., 0]

    if location == "gauss":
        return gauss_to_nodal(scalar, vd)
    if location == "element":
        return element_to_nodal(scalar, vd)

    n_nodes = int(vd.points.shape[0]) if vd.points is not None else int(np.asarray(scalar).shape[0])
    scalar = np.asarray(scalar, dtype=np.float64).reshape(-1)
    if scalar.shape[0] == n_nodes:
        return scalar
    if scalar.size == n_nodes:
        return scalar.reshape(n_nodes)
    if n_nodes > 0 and scalar.size % n_nodes == 0:
        ndof = scalar.size // n_nodes
        return np.linalg.norm(scalar.reshape(n_nodes, ndof), axis=-1)
    if scalar.shape[0] > n_nodes:
        return scalar[:n_nodes]
    return np.pad(scalar, (0, max(0, n_nodes - scalar.shape[0])))


def attach_field(
    grid: "pv.UnstructuredGrid",
    field: np.ndarray,
    name: str,
    vd: VizData,
    location: str = "gauss",
    *,
    component: Optional[int] = None,
    field_key: Optional[str] = None,
) -> None:
    """Attach a scalar field to *grid* for rendering.

    For Gauss-located fields, the values are first averaged to nodes.
    Multi-component fields are reduced to a single component or to
    von Mises (for 6-component stress/strain).

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
    field : ndarray
        Raw field data (single frame).
    name : str
        Array name to attach.
    vd : VizData
    location : str
        ``"gauss"`` or ``"node"``.
    component : int, optional
        Which component to extract.  ``None`` means magnitude / von Mises.
    """
    scalar = scalarize_field(
        field,
        vd,
        location=location,
        component=component,
        field_key=field_key,
    )
    scalar = np.asarray(scalar, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scalar)
    if not np.all(finite):
        fill = float(np.nanmean(scalar[finite])) if np.any(finite) else 0.0
        scalar = np.where(finite, scalar, fill)
    if scalar.shape[0] != grid.n_points:
        scalar = scalar[:grid.n_points] if scalar.shape[0] > grid.n_points else np.pad(
            scalar,
            (0, grid.n_points - scalar.shape[0]),
        )
    grid.point_data[name] = scalar.astype(np.float64)


def _von_mises_6(s: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress for 6-component Voigt vector.

    Components: s11, s22, s33, s12, s13, s23.
    """
    s11, s22, s33 = s[..., 0], s[..., 1], s[..., 2]
    s12, s13, s23 = s[..., 3], s[..., 4], s[..., 5]
    vm = np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
        + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    )
    return vm


def _von_mises_3(s: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress for plane-stress Voigt vectors."""
    s11, s22, s12 = s[..., 0], s[..., 1], s[..., 2]
    return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3.0 * s12 ** 2)


def _equivalent_strain_6(eps: np.ndarray) -> np.ndarray:
    """Small-strain equivalent strain using engineering shear components."""
    e11, e22, e33 = eps[..., 0], eps[..., 1], eps[..., 2]
    g12, g13, g23 = eps[..., 3], eps[..., 4], eps[..., 5]
    diff_sq = (e11 - e22) ** 2 + (e22 - e33) ** 2 + (e33 - e11) ** 2
    shear_sq = g12 ** 2 + g13 ** 2 + g23 ** 2
    return np.sqrt((2.0 / 9.0) * diff_sq + (1.0 / 3.0) * shear_sq)


def _equivalent_strain_3(eps: np.ndarray) -> np.ndarray:
    """Plane equivalent strain using engineering shear ``gamma12``."""
    e11, e22, g12 = eps[..., 0], eps[..., 1], eps[..., 2]
    return np.sqrt((4.0 / 9.0) * (e11 ** 2 - e11 * e22 + e22 ** 2) + (1.0 / 3.0) * g12 ** 2)


def _classify_field_kind(field_key: Optional[str]) -> str:
    key = str(field_key or "").lower()
    if "peeq" in key:
        return "scalar"
    if "generalized_stress" in key or "generalized_strain" in key:
        return "vector"
    if key.startswith(("rsdms_sf_", "rsdms_sm_", "rsdms_ge_", "rsdms_gk_")):
        return "vector"
    if "stress" in key or key.startswith(("rsdms_s_res", "rsdms_s_tot", "rsdm_s_res", "rsdm_s_tot", "rsdm_xs_", "rsdms_xs_")):
        return "stress"
    if "strain" in key or key.startswith(("rsdms_e_res", "rsdms_e_tot", "rsdm_e_res", "rsdm_e_tot")):
        return "strain"
    if key in {"u", "frame_u", "elastic_u", "solid_u_nodal"} or "displacement" in key:
        return "vector"
    if (
        "nforc" in key
        or "reaction" in key
        or "internal_force" in key
        or "force" in key
        or key in {"rsdms_ceq", "rsdms_ferror", "rsdm_ceq", "rsdm_ferror"}
    ):
        return "vector"
    return "other"
