"""
Native surface-load assembly helpers for solid elastic models.

These helpers operate only on the parsed input payload already produced by the
ABAQUS parser. They do not reopen ``.inp`` files and remain fully inside the
native ``jaxmech`` stack.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from jaxmech.fem.mesh import ElementBlock


SUPPORTED_SURFACE_LOAD_ELE_TYPES = {
    "C3D8",
    "C3D8R",
    "C3D6",
    "C3D4",
    "CPS4",
    "CPS4R",
    "CPS3",
    "CPE4",
    "CPE4R",
    "CPE3",
}

ABAQUS_FACE_INDS_BY_ELE_TYPE = {
    "CPS4": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPS4R": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPE4": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPE4R": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 3], dtype=np.int32),
        "S4": np.array([3, 0], dtype=np.int32),
    },
    "CPS3": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 0], dtype=np.int32),
    },
    "CPE3": {
        "S1": np.array([0, 1], dtype=np.int32),
        "S2": np.array([1, 2], dtype=np.int32),
        "S3": np.array([2, 0], dtype=np.int32),
    },
    "C3D8": {
        "S1": np.array([0, 1, 2, 3], dtype=np.int32),
        "S2": np.array([4, 5, 6, 7], dtype=np.int32),
        "S3": np.array([0, 1, 5, 4], dtype=np.int32),
        "S4": np.array([1, 2, 6, 5], dtype=np.int32),
        "S5": np.array([2, 3, 7, 6], dtype=np.int32),
        "S6": np.array([3, 0, 4, 7], dtype=np.int32),
    },
    "C3D8R": {
        "S1": np.array([0, 1, 2, 3], dtype=np.int32),
        "S2": np.array([4, 5, 6, 7], dtype=np.int32),
        "S3": np.array([0, 1, 5, 4], dtype=np.int32),
        "S4": np.array([1, 2, 6, 5], dtype=np.int32),
        "S5": np.array([2, 3, 7, 6], dtype=np.int32),
        "S6": np.array([3, 0, 4, 7], dtype=np.int32),
    },
    "C3D6": {
        "S1": np.array([0, 1, 2], dtype=np.int32),
        "S2": np.array([3, 4, 5], dtype=np.int32),
        "S3": np.array([0, 1, 4, 3], dtype=np.int32),
        "S4": np.array([1, 2, 5, 4], dtype=np.int32),
        "S5": np.array([2, 0, 3, 5], dtype=np.int32),
    },
    "C3D4": {
        "S1": np.array([0, 1, 2], dtype=np.int32),
        "S2": np.array([0, 3, 1], dtype=np.int32),
        "S3": np.array([1, 3, 2], dtype=np.int32),
        "S4": np.array([2, 3, 0], dtype=np.int32),
    },
}


def _unwrap_string(value) -> str:
    return str(np.asarray(value).reshape(-1)[0]).strip().upper()


def _as_int_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.int32).reshape(-1)


def _normalize_element_set_blocks(parsed_data: dict) -> dict[str, list[dict]]:
    if "element_set_blocks" in parsed_data:
        normalized: dict[str, list[dict]] = {}
        for set_name, records in parsed_data["element_set_blocks"].items():
            normalized[str(set_name).upper()] = [
                {
                    "block_id": int(record["block_id"]),
                    "cell_type": str(record["cell_type"]),
                    "ele_type": str(record["ele_type"]).upper(),
                    "elem_ids": _as_int_vector(record["elem_ids"]),
                }
                for record in records
            ]
        return normalized

    normalized = {}
    if "element_set_block_record_names" not in parsed_data:
        return normalized
    record_names = [_unwrap_string(x) for x in parsed_data["element_set_block_record_names"].reshape(-1)]
    record_ids = np.asarray(parsed_data["element_set_block_record_ids"], dtype=np.int32).reshape(-1)
    record_types = [_unwrap_string(x) for x in parsed_data["element_set_block_record_types"].reshape(-1)]
    etype_key = (
        "element_set_block_record_etype"
        if "element_set_block_record_etype" in parsed_data
        else "element_set_block_record_ele_types"
    )
    record_ele_types = [_unwrap_string(x) for x in parsed_data[etype_key].reshape(-1)]
    record_elems = parsed_data["element_set_block_record_elems"].reshape(-1)
    for set_name, block_id, cell_type, ele_type, elems in zip(
        record_names, record_ids, record_types, record_ele_types, record_elems
    ):
        normalized.setdefault(set_name, []).append(
            {
                "block_id": int(block_id),
                "cell_type": cell_type,
                "ele_type": ele_type,
                "elem_ids": _as_int_vector(elems),
            }
        )
    return normalized


def _normalize_surface_defs(parsed_data: dict) -> dict[str, list[tuple[str, str]]]:
    if "surface_defs" in parsed_data:
        return {
            str(name).upper(): [(str(elset).upper(), str(face).upper()) for elset, face in items]
            for name, items in parsed_data["surface_defs"].items()
        }

    normalized = {}
    if "surface_names" not in parsed_data:
        return normalized
    names = [_unwrap_string(x) for x in parsed_data["surface_names"].reshape(-1)]
    items = parsed_data["surface_items"].reshape(-1)
    for idx, name in enumerate(names):
        surface_items = []
        for rec in list(items[idx]):
            arr = np.asarray(rec).reshape(-1)
            if arr.size >= 2:
                surface_items.append((str(arr[0]).strip().upper(), str(arr[1]).strip().upper()))
        normalized[name] = surface_items
    return normalized


def _get_face_node_indices(ele_type: str, face_label: str) -> np.ndarray:
    ele_type = str(ele_type).upper()
    face_map = ABAQUS_FACE_INDS_BY_ELE_TYPE.get(ele_type)
    if face_map is None:
        raise NotImplementedError(f"Unsupported element type for surface load integration: {ele_type}")
    local_face_nodes = face_map.get(str(face_label).upper())
    if local_face_nodes is None:
        raise NotImplementedError(f"Unsupported face label '{face_label}' for element type '{ele_type}'.")
    return local_face_nodes


def _edge_rule_line2() -> tuple[np.ndarray, np.ndarray]:
    a = 1.0 / np.sqrt(3.0)
    return np.asarray([-a, a], dtype=np.float64), np.ones((2,), dtype=np.float64)


def _integrate_constant_traction_on_edge(edge_coords: np.ndarray, traction: np.ndarray) -> np.ndarray:
    edge_coords = np.asarray(edge_coords, dtype=np.float64)
    if edge_coords.shape != (2, 2):
        raise NotImplementedError(f"Unsupported edge coordinates shape: {edge_coords.shape}")
    traction = np.asarray(traction, dtype=np.float64).reshape(1, 2)
    quad_points, quad_weights = _edge_rule_line2()
    local_force = np.zeros((2, 2), dtype=np.float64)
    for qp, weight in zip(quad_points, quad_weights):
        shape_vals = np.asarray([0.5 * (1.0 - qp), 0.5 * (1.0 + qp)], dtype=np.float64)
        shape_grads = np.asarray([-0.5, 0.5], dtype=np.float64)
        tangent = shape_grads @ edge_coords
        line_scale = np.linalg.norm(tangent)
        local_force += shape_vals[:, None] * traction * (float(weight) * line_scale)
    return local_force


def _face_rule_quad4() -> tuple[np.ndarray, np.ndarray]:
    a = 1.0 / np.sqrt(3.0)
    points = np.asarray([[-a, -a], [a, -a], [a, a], [-a, a]], dtype=np.float64)
    weights = np.ones((4,), dtype=np.float64)
    return points, weights


def _face_rule_tri3() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64), np.asarray([0.5], dtype=np.float64)


def _quad4_face_shape(xi_eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = float(xi_eta[0])
    eta = float(xi_eta[1])
    shape_vals = np.asarray(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dshape_dxi = np.asarray(
        [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dshape_deta = np.asarray(
        [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
        dtype=np.float64,
    )
    return shape_vals, dshape_dxi, dshape_deta


def _tri3_face_shape(xi_eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = float(xi_eta[0])
    eta = float(xi_eta[1])
    shape_vals = np.asarray([1.0 - xi - eta, xi, eta], dtype=np.float64)
    dshape_dxi = np.asarray([-1.0, 1.0, 0.0], dtype=np.float64)
    dshape_deta = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    return shape_vals, dshape_dxi, dshape_deta


def _integrate_constant_traction_on_face(face_coords: np.ndarray, traction: np.ndarray) -> np.ndarray:
    face_coords = np.asarray(face_coords, dtype=np.float64)
    traction = np.asarray(traction, dtype=np.float64).reshape(1, 3)
    if face_coords.shape[0] == 4:
        quad_points, quad_weights = _face_rule_quad4()
        shape_rule = _quad4_face_shape
    elif face_coords.shape[0] == 3:
        quad_points, quad_weights = _face_rule_tri3()
        shape_rule = _tri3_face_shape
    else:
        raise NotImplementedError(f"Unsupported face node count: {face_coords.shape[0]}")

    local_force = np.zeros((face_coords.shape[0], 3), dtype=np.float64)
    for qp, weight in zip(quad_points, quad_weights):
        shape_vals, dshape_dxi, dshape_deta = shape_rule(qp)
        tangent_1 = dshape_dxi @ face_coords
        tangent_2 = dshape_deta @ face_coords
        area_scale = np.linalg.norm(np.cross(tangent_1, tangent_2))
        local_force += shape_vals[:, None] * traction * (float(weight) * area_scale)
    return local_force


def _iter_pressure_surface_loads(dsload_entries: Iterable[tuple[str, str, float]]):
    for surf_name, load_type, magnitude in dsload_entries:
        if str(load_type).upper() != "P":
            continue
        yield str(surf_name).upper(), float(magnitude)


def assemble_surface_load_mixed(
    points: np.ndarray,
    blocks: list[ElementBlock],
    parsed_data: dict,
    dsload_entries: list[tuple[str, str, float]],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    vec = int(points.shape[1])
    if vec not in (2, 3):
        raise NotImplementedError(f"Unsupported spatial dimension for surface loads: {vec}")

    blocks_by_id = {int(block.block_id): block for block in blocks}
    element_set_blocks = _normalize_element_set_blocks(parsed_data)
    surface_defs = _normalize_surface_defs(parsed_data)
    f_ext = np.zeros((points.shape[0], vec), dtype=np.float64)

    for surf_name, p_abaqus in _iter_pressure_surface_loads(dsload_entries):
        surface_items = surface_defs.get(surf_name, [])
        if not surface_items:
            continue

        traction_scale = -float(p_abaqus)
        for elset_name, face_label in surface_items:
            for record in element_set_blocks.get(elset_name.upper(), []):
                ele_type = str(record["ele_type"]).upper()
                if ele_type not in SUPPORTED_SURFACE_LOAD_ELE_TYPES:
                    raise NotImplementedError(
                        f"Surface load assembly currently supports {sorted(SUPPORTED_SURFACE_LOAD_ELE_TYPES)} only; "
                        f"got {ele_type} on elset '{elset_name}'."
                    )
                block = blocks_by_id.get(int(record["block_id"]))
                if block is None:
                    raise KeyError(f"Missing element block {record['block_id']} referenced by '{elset_name}'.")
                local_face_nodes = _get_face_node_indices(ele_type, face_label)
                connectivity = np.asarray(block.connectivity, dtype=np.int32)
                for elem_id in _as_int_vector(record["elem_ids"]):
                    conn = connectivity[int(elem_id)]
                    face_node_ids = conn[local_face_nodes]
                    if vec == 2:
                        elem_coords = points[conn, :2]
                        edge_coords = points[face_node_ids, :2]
                        elem_centroid = np.mean(elem_coords, axis=0)
                        edge_centroid = np.mean(edge_coords, axis=0)
                        tangent = edge_coords[1] - edge_coords[0]
                        tangent_norm = np.linalg.norm(tangent)
                        if tangent_norm < 1e-12:
                            continue
                        normal = np.asarray([tangent[1], -tangent[0]], dtype=np.float64) / tangent_norm
                        if np.dot(normal, edge_centroid - elem_centroid) < 0.0:
                            normal = -normal
                        traction = traction_scale * normal
                        f_ext[face_node_ids] += _integrate_constant_traction_on_edge(edge_coords, traction)
                    else:
                        elem_coords = points[conn]
                        face_coords = points[face_node_ids]
                        elem_centroid = np.mean(elem_coords, axis=0)
                        face_centroid = np.mean(face_coords, axis=0)
                        tangent_1 = face_coords[1] - face_coords[0]
                        tangent_2 = face_coords[2] - face_coords[0]
                        normal = np.cross(tangent_1, tangent_2)
                        normal_norm = np.linalg.norm(normal)
                        if normal_norm < 1e-12:
                            continue
                        normal = normal / normal_norm
                        if np.dot(normal, face_centroid - elem_centroid) < 0.0:
                            normal = -normal
                        traction = traction_scale * normal
                        f_ext[face_node_ids] += _integrate_constant_traction_on_face(face_coords, traction)

    return f_ext.reshape(-1)
