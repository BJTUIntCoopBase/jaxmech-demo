#!/usr/bin/env python
# -*- coding: ascii -*-
"""
export_unified_odb.py
=====================

Unified Abaqus ODB extractor for this project.

One MAT file is written per ODB and contains both:
1. Elastic-comparison fields used by ``compare_elastic_results.py``
2. Legacy shakedown fields used by ``Extract3DConvertProcess.m``

The project-internal stress/strain ordering is always:
    [11, 22, 33, 12, 23, 13]
and C3D8 integration points are always reordered to the JAX convention.
"""

from __future__ import print_function

import argparse
import datetime
import os

import numpy as np


JAX_FROM_ABA = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)
VOIGT_PERM_3D = np.array([0, 1, 2, 3, 5, 4], dtype=np.int32)


def _to_float64(data):
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _assign_vector_prefix(dst, src):
    src_arr = _to_float64(src).reshape(-1)
    n = min(int(dst.shape[-1]), int(src_arr.shape[0]))
    if n > 0:
        dst[..., :n] = src_arr[:n]


def _add_vector_prefix(dst, src):
    src_arr = _to_float64(src).reshape(-1)
    n = min(int(dst.shape[-1]), int(src_arr.shape[0]))
    if n > 0:
        dst[..., :n] += src_arr[:n]


def _get_last_key(mapping):
    keys = list(mapping.keys())
    if not keys:
        raise RuntimeError("Expected a non-empty Abaqus mapping.")
    return keys[-1]


def _find_instance(odb, instance_name=None):
    instances = odb.rootAssembly.instances
    if not instances:
        raise RuntimeError("No instances found in ODB root assembly.")

    if instance_name is None:
        return instances[sorted(instances.keys())[0]]

    key = instance_name.upper()
    if key in instances:
        return instances[key]
    for name, inst in instances.items():
        if name.upper() == instance_name.upper():
            return inst
    raise KeyError("Instance '{}' not found in ODB.".format(instance_name))


def _find_element_set(container, set_name):
    if set_name is None:
        return None
    if hasattr(container, "elementSets") and set_name in container.elementSets:
        return container.elementSets[set_name]
    if hasattr(container, "elementSets"):
        for name, element_set in container.elementSets.items():
            if name.upper() == set_name.upper():
                return element_set
    return None


def _find_node_set(root_assembly, set_name):
    if not set_name:
        return []
    if set_name in root_assembly.nodeSets:
        return root_assembly.nodeSets[set_name].nodes
    for name, node_set in root_assembly.nodeSets.items():
        if name.upper() == set_name.upper():
            return node_set.nodes
    raise KeyError("Node set '{}' not found in ODB root assembly.".format(set_name))


def _node_set_labels(node_groups):
    labels = []
    for group in node_groups:
        for node in group:
            labels.append(int(node.label))
    return sorted(set(labels))


def _iter_element_labels(elements_obj):
    if elements_obj is None:
        return
    try:
        for element in elements_obj:
            if hasattr(element, "label"):
                yield int(element.label)
            else:
                for sub in element:
                    if hasattr(sub, "label"):
                        yield int(sub.label)
    except TypeError:
        return


def _collect_node_set_label_map(root_assembly, inst):
    label_map = {}

    def _store(name, node_groups):
        if not name:
            return
        try:
            labels = _node_set_labels(node_groups)
        except Exception:
            return
        label_map[name.upper()] = labels

    if hasattr(root_assembly, "nodeSets"):
        for name, node_set in root_assembly.nodeSets.items():
            _store(name, getattr(node_set, "nodes", []))
    if hasattr(inst, "nodeSets"):
        for name, node_set in inst.nodeSets.items():
            _store(name, getattr(node_set, "nodes", []))
    return label_map


def _region_node_labels(region):
    if region is None:
        return []
    if hasattr(region, "nodes"):
        try:
            return _node_set_labels(region.nodes)
        except Exception:
            pass
    if hasattr(region, "name"):
        return None
    return []


def _is_constrained_component(value):
    if value is None:
        return False
    text = str(value).strip().upper()
    if text in ("", "UNSET", "FREED", "UNKNOWN"):
        return False
    return True


def _collect_bc_constrained_nodes(odb, root_assembly, inst, upto_step_name=None):
    node_set_map = _collect_node_set_label_map(root_assembly, inst)
    dof_labels = {1: set(), 2: set(), 3: set()}

    for step_name, step in odb.steps.items():
        if hasattr(step, "boundaryConditions"):
            for _, bc in step.boundaryConditions.items():
                region = getattr(bc, "region", None)
                labels = _region_node_labels(region)
                if labels is None:
                    region_name = getattr(region, "name", "")
                    labels = node_set_map.get(str(region_name).upper(), [])
                for dof, attr in ((1, "u1"), (2, "u2"), (3, "u3")):
                    if _is_constrained_component(getattr(bc, attr, None)):
                        dof_labels[dof].update(labels)
        if upto_step_name is not None and str(step_name) == str(upto_step_name):
            break

    return (
        sorted(dof_labels[1]),
        sorted(dof_labels[2]),
        sorted(dof_labels[3]),
    )


def _ip_index_0based(value):
    ip = getattr(value, "integrationPoint", None)
    if ip is None:
        return 0
    if hasattr(ip, "number"):
        return int(ip.number) - 1
    if isinstance(ip, int):
        return int(ip) - 1
    return 0


def _reorder_ip_data(ele_type, values):
    data = [list(row) for row in values]
    if ele_type == "C3D8" and len(data) >= 8:
        return [data[i] for i in JAX_FROM_ABA.tolist()]
    if ele_type in ("CPS4", "CPE4") and len(data) >= 4:
        point4 = data.pop(2)
        data.insert(3, point4)
    return data


def _reorder_voigt_data(ele_type, values):
    data = [list(row) for row in values]
    if data and len(data[0]) == 6 and not (ele_type.startswith("CPS") or ele_type.startswith("CPE")):
        return [[row[i] for i in VOIGT_PERM_3D.tolist()] for row in data]
    return data


def _filter_2d_voigt(ele_type, values):
    if ele_type.startswith("CPS") or ele_type.startswith("CPE"):
        return [[row[0], row[1], row[3]] for row in values]
    return values


def _ip_sorted_values(field_subset):
    values = list(field_subset.values)
    values.sort(key=_ip_index_0based)
    rows = []
    for value in values:
        data = _to_float64(value.data).reshape(-1)
        rows.append([float(item) for item in data])
    return rows


def _stack_or_object(data, dtype):
    try:
        return np.asarray(data, dtype=dtype)
    except Exception:
        return np.asarray(data, dtype=object)


def _build_matlab_scalar_cell_1d(data):
    arr = np.empty((len(data), 1), dtype=object)
    for i, value in enumerate(data):
        arr[i, 0] = value
    return arr


def _build_matlab_cell_nd(data):
    dense = np.asarray(data)
    cell = np.empty(dense.shape, dtype=object)
    for index in np.ndindex(dense.shape):
        cell[index] = dense[index].item() if hasattr(dense[index], "item") else dense[index]
    return cell


def _parse_material_map(items):
    mapping = []
    for item in items:
        if ":" not in item:
            raise ValueError(
                "Invalid --material-map '{}'. Expected format ELEMENT_SET:MATERIAL_ID.".format(item)
            )
        set_name, mat_id = item.split(":", 1)
        mapping.append((set_name.strip(), int(mat_id.strip())))
    return mapping


def _resolve_output_mat_path(odb_path, out_prefix=None, out_dir=None):
    odb_stem = os.path.splitext(os.path.basename(odb_path))[0]
    if out_prefix:
        return os.path.abspath(out_prefix) + ".mat"
    if out_dir:
        return os.path.join(os.path.abspath(out_dir), odb_stem + ".mat")
    return os.path.splitext(os.path.abspath(odb_path))[0] + ".mat"


def export_odb(
    odb_path,
    out_prefix=None,
    out_dir=None,
    instance_name=None,
    step_name=None,
    frame_index=-1,
    skip_initial=False,
    fixed_dof1_set=None,
    fixed_dof2_set=None,
    fixed_dof3_set=None,
    material_map=None,
    default_material_id=1,
):
    try:
        from odbAccess import openOdb
        from abaqusConstants import INTEGRATION_POINT, NODAL
        import scipy.io as sio
    except ImportError:
        raise SystemExit(
            "Run this script with Abaqus Python and ensure scipy.io is available."
        )

    if not os.path.isfile(odb_path):
        raise SystemExit("ODB not found: {}".format(odb_path))

    odb = openOdb(path=odb_path, readOnly=True)
    try:
        inst = _find_instance(odb, instance_name)

        nodes_sorted = sorted(inst.nodes, key=lambda n: n.label)
        node_labels = np.array([n.label for n in nodes_sorted], dtype=np.int32)
        n_nodes = len(node_labels)
        node_index = {lbl: i for i, lbl in enumerate(node_labels)}
        node_coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for nd in nodes_sorted:
            node_coords[node_index[nd.label], :] = nd.coordinates

        elems_sorted = sorted(inst.elements, key=lambda e: e.label)
        elem_labels = np.array([e.label for e in elems_sorted], dtype=np.int32)

        if step_name:
            steps_iter = [(step_name, odb.steps[step_name])]
        else:
            steps_iter = list(odb.steps.items())

        frame_records = []
        for step_key, step in steps_iter:
            for frame_local_idx, frame in enumerate(step.frames):
                if skip_initial and float(frame.frameValue) == 0.0:
                    continue
                fo = frame.fieldOutputs

                U_data = np.zeros((n_nodes, 3), dtype=np.float64)
                if "U" in fo:
                    subset = fo["U"].getSubset(region=inst, position=NODAL)
                    if not subset.values:
                        subset = fo["U"].getSubset(position=NODAL)
                    for val in subset.values:
                        if val.nodeLabel in node_index:
                            _assign_vector_prefix(U_data[node_index[val.nodeLabel], :], val.data)

                NFORC_data = np.zeros((n_nodes, 3), dtype=np.float64)
                if "NFORC" in fo:
                    field_obj = fo["NFORC"]
                    subset = field_obj.getSubset(region=inst, position=NODAL)
                    if not subset.values:
                        subset = field_obj.getSubset(position=NODAL)
                    values = subset.values if subset.values else field_obj.values
                    for val in values:
                        if val.nodeLabel in node_index:
                            _add_vector_prefix(NFORC_data[node_index[val.nodeLabel], :], val.data)
                else:
                    for ci, fname in enumerate(["NFORC1", "NFORC2", "NFORC3"]):
                        try:
                            field_obj = fo[fname]
                        except Exception:
                            continue
                        subset = field_obj.getSubset(region=inst, position=NODAL)
                        if not subset.values:
                            subset = field_obj.getSubset(position=NODAL)
                        values = subset.values if subset.values else field_obj.values
                        for val in values:
                            if val.nodeLabel in node_index:
                                NFORC_data[node_index[val.nodeLabel], ci] += _to_float64(val.data)[0]

                peeq_values = []
                if "PEEQ" in fo:
                    field_obj = fo["PEEQ"]
                    subset = field_obj.getSubset(region=inst, position=INTEGRATION_POINT)
                    values = subset.values if subset.values else field_obj.values
                    for val in values:
                        peeq_data = _to_float64(val.data).reshape(-1)
                        if peeq_data.size:
                            peeq_values.append(float(peeq_data[0]))

                frame_stress_flat = np.empty((0, 0), dtype=np.float64)
                frame_strain_flat = np.empty((0, 0), dtype=np.float64)
                frame_peeq_flat = np.empty((0,), dtype=np.float64)
                if "S" in fo and "E" in fo:
                    stress_res_frame = fo["S"].getSubset(region=inst, position=INTEGRATION_POINT)
                    strain_res_frame = fo["E"].getSubset(region=inst, position=INTEGRATION_POINT)
                    peeq_res_frame = None
                    if "PEEQ" in fo:
                        peeq_res_frame = fo["PEEQ"].getSubset(region=inst, position=INTEGRATION_POINT)

                    stress_by_type = {}
                    strain_by_type = {}
                    peeq_by_type = {}
                    for element in elems_sorted:
                        ele_type = str(element.type)
                        stress_vals = _ip_sorted_values(stress_res_frame.getSubset(region=element))
                        strain_vals = _ip_sorted_values(strain_res_frame.getSubset(region=element))
                        stress_vals = _reorder_ip_data(ele_type, stress_vals)
                        strain_vals = _reorder_ip_data(ele_type, strain_vals)
                        stress_vals = _reorder_voigt_data(ele_type, stress_vals)
                        strain_vals = _reorder_voigt_data(ele_type, strain_vals)
                        stress_vals = _filter_2d_voigt(ele_type, stress_vals)
                        strain_vals = _filter_2d_voigt(ele_type, strain_vals)
                        if peeq_res_frame is not None:
                            peeq_vals = _ip_sorted_values(peeq_res_frame.getSubset(region=element))
                            peeq_vals = _reorder_ip_data(ele_type, peeq_vals)
                            peeq_vals = [float(np.asarray(row, dtype=np.float64).reshape(-1)[0]) for row in peeq_vals]
                        else:
                            peeq_vals = [0.0] * len(stress_vals)

                        stress_by_type.setdefault(ele_type, []).append(np.asarray(stress_vals, dtype=np.float64))
                        strain_by_type.setdefault(ele_type, []).append(np.asarray(strain_vals, dtype=np.float64))
                        peeq_by_type.setdefault(ele_type, []).append(np.asarray(peeq_vals, dtype=np.float64))

                    if stress_by_type:
                        frame_stress_flat = np.concatenate(
                            [np.asarray(stress_by_type[ele_type], dtype=np.float64).reshape(-1, np.asarray(stress_by_type[ele_type], dtype=np.float64).shape[-1]) for ele_type in sorted(stress_by_type)],
                            axis=0,
                        )
                        frame_strain_flat = np.concatenate(
                            [np.asarray(strain_by_type[ele_type], dtype=np.float64).reshape(-1, np.asarray(strain_by_type[ele_type], dtype=np.float64).shape[-1]) for ele_type in sorted(strain_by_type)],
                            axis=0,
                        )
                        frame_peeq_flat = np.concatenate(
                            [np.asarray(peeq_by_type[ele_type], dtype=np.float64).reshape(-1) for ele_type in sorted(peeq_by_type)],
                            axis=0,
                        )

                frame_records.append(
                    {
                        "step_name": step_key,
                        "frame_index_local": int(frame_local_idx),
                        "frame_value": float(frame.frameValue),
                        "U": U_data,
                        "NFORC": NFORC_data,
                        "gauss_stress": frame_stress_flat,
                        "gauss_strain": frame_strain_flat,
                        "gauss_peeq": frame_peeq_flat,
                        "PEEQ_max": float(np.max(peeq_values)) if peeq_values else 0.0,
                        "PEEQ_mean": float(np.mean(peeq_values)) if peeq_values else 0.0,
                    }
                )

        if not frame_records:
            raise RuntimeError("No frames were extracted from ODB.")

        # Select final frame for legacy fields.
        legacy_frame = frame_records[frame_index]
        final_step_name = legacy_frame["step_name"]
        final_frame_value = legacy_frame["frame_value"]

        final_step = odb.steps[final_step_name]
        final_frame = final_step.frames[frame_index]
        disp_res = final_frame.fieldOutputs["U"].getSubset(region=inst, position=NODAL)
        stress_res = final_frame.fieldOutputs["S"].getSubset(region=inst, position=INTEGRATION_POINT)
        strain_res = final_frame.fieldOutputs["E"].getSubset(region=inst, position=INTEGRATION_POINT)
        peeq_res = None
        if "PEEQ" in final_frame.fieldOutputs:
            peeq_res = final_frame.fieldOutputs["PEEQ"].getSubset(region=inst, position=INTEGRATION_POINT)

        node_disp = {}
        for value in disp_res.values:
            node_disp[int(value.nodeLabel)] = list(map(float, value.data))

        node_by_label = {int(node.label): node for node in inst.nodes}
        element_material_ids = {}
        material_map = [] if material_map is None else list(material_map)
        for set_name, mat_id in material_map:
            element_set = _find_element_set(inst, set_name)
            if element_set is None:
                assembly_set = _find_element_set(odb.rootAssembly, set_name)
                if assembly_set is None:
                    raise KeyError("Element set '{}' not found.".format(set_name))
                element_set = assembly_set
            for element_label in _iter_element_labels(getattr(element_set, "elements", None)):
                element_material_ids[int(element_label)] = int(mat_id)

        element_types = []
        element_connectivity = []
        element_node_coords = []
        element_node_disp = []
        element_gauss_stress = []
        element_gauss_strain = []
        element_gauss_peeq = []
        element_material_id_list = []

        for element in elems_sorted:
            ele_label = int(element.label)
            ele_type = str(element.type)
            conn = [int(label) for label in element.connectivity]

            coords = [list(map(float, node_by_label[label].coordinates)) for label in conn]
            disp = [node_disp.get(label, [0.0, 0.0, 0.0]) for label in conn]

            stress_vals = _ip_sorted_values(stress_res.getSubset(region=element))
            strain_vals = _ip_sorted_values(strain_res.getSubset(region=element))
            stress_vals = _reorder_ip_data(ele_type, stress_vals)
            strain_vals = _reorder_ip_data(ele_type, strain_vals)
            stress_vals = _reorder_voigt_data(ele_type, stress_vals)
            strain_vals = _reorder_voigt_data(ele_type, strain_vals)
            stress_vals = _filter_2d_voigt(ele_type, stress_vals)
            strain_vals = _filter_2d_voigt(ele_type, strain_vals)
            if peeq_res is not None:
                peeq_vals = _ip_sorted_values(peeq_res.getSubset(region=element))
                peeq_vals = _reorder_ip_data(ele_type, peeq_vals)
                peeq_vals = [float(np.asarray(row, dtype=np.float64).reshape(-1)[0]) for row in peeq_vals]
            else:
                peeq_vals = [0.0] * len(stress_vals)

            element_types.append(ele_type)
            element_connectivity.append(conn)
            element_node_coords.append(coords)
            element_node_disp.append(disp)
            element_gauss_stress.append(stress_vals)
            element_gauss_strain.append(strain_vals)
            element_gauss_peeq.append(peeq_vals)
            element_material_id_list.append(element_material_ids.get(ele_label, int(default_material_id)))

        frame_times = np.asarray([rec["frame_value"] for rec in frame_records], dtype=np.float64)
        frame_step_names = np.asarray([str(rec["step_name"]) for rec in frame_records], dtype=object)
        frame_index_local = np.asarray([rec["frame_index_local"] for rec in frame_records], dtype=np.int32)
        frame_u_all = np.stack([rec["U"] for rec in frame_records], axis=0)
        NFORC_all = np.stack([rec["NFORC"] for rec in frame_records], axis=0)
        frame_gauss_stress_all = np.stack([rec["gauss_stress"] for rec in frame_records], axis=0)
        frame_gauss_strain_all = np.stack([rec["gauss_strain"] for rec in frame_records], axis=0)
        frame_gauss_peeq_all = np.stack([rec["gauss_peeq"] for rec in frame_records], axis=0)
        peeq_time_max = np.asarray([rec["PEEQ_max"] for rec in frame_records], dtype=np.float64)
        peeq_time_mean = np.asarray([rec["PEEQ_mean"] for rec in frame_records], dtype=np.float64)

        info_str = (
            "Unified export from {} | Date {} | "
            "IP order: JAX | Voigt: [S11,S22,S33,S12,S23,S13]"
        ).format(odb_path, datetime.datetime.now().isoformat())

        node_dof1_labels, node_dof2_labels, node_dof3_labels = _collect_bc_constrained_nodes(
            odb,
            odb.rootAssembly,
            inst,
            upto_step_name=final_step_name,
        )
        if fixed_dof1_set:
            try:
                node_dof1_labels = sorted(set(node_dof1_labels).union(
                    _node_set_labels(_find_node_set(odb.rootAssembly, fixed_dof1_set))
                ))
            except Exception:
                pass
        if fixed_dof2_set:
            try:
                node_dof2_labels = sorted(set(node_dof2_labels).union(
                    _node_set_labels(_find_node_set(odb.rootAssembly, fixed_dof2_set))
                ))
            except Exception:
                pass
        if fixed_dof3_set:
            try:
                node_dof3_labels = sorted(set(node_dof3_labels).union(
                    _node_set_labels(_find_node_set(odb.rootAssembly, fixed_dof3_set))
                ))
            except Exception:
                pass

        ele_label_set = _build_matlab_scalar_cell_1d(elem_labels.tolist())
        ele_type_set = _build_matlab_scalar_cell_1d(element_types)
        ele_material_id_set = _build_matlab_scalar_cell_1d(element_material_id_list)
        node_constrained_dof1 = _build_matlab_scalar_cell_1d(node_dof1_labels)
        node_constrained_dof2 = _build_matlab_scalar_cell_1d(node_dof2_labels)
        node_constrained_dof3 = _build_matlab_scalar_cell_1d(node_dof3_labels)
        ele_node_label_set = _build_matlab_cell_nd(element_connectivity)
        ele_node_coord_set = _build_matlab_cell_nd(element_node_coords)
        ele_node_disp_set = _build_matlab_cell_nd(element_node_disp)
        ele_gauss_stress_set = _build_matlab_cell_nd(element_gauss_stress)
        ele_gauss_strain_set = _build_matlab_cell_nd(element_gauss_strain)
        ele_gauss_peeq_set = _build_matlab_cell_nd(element_gauss_peeq)

        save_dict = {
            # Elastic compare fields
            "info": np.array(info_str),
            "odb_path": np.array(odb_path),
            "frame_time": frame_times,
            "frame_step_name": frame_step_names,
            "frame_index_local": frame_index_local,
            "frame_u": frame_u_all,
            "frame_nforc": NFORC_all,
            "frame_gauss_stress": frame_gauss_stress_all,
            "frame_gauss_strain": frame_gauss_strain_all,
            "frame_gauss_peeq": frame_gauss_peeq_all,
            "voigt_order": np.array("S11,S22,S33,S12,S23,S13"),
            "ip_order": np.array("JAX"),
            "jax_from_aba": JAX_FROM_ABA,
            "voigt_perm": VOIGT_PERM_3D,
            "NFORC": NFORC_all,
            "PEEQ_time_max": peeq_time_max,
            "PEEQ_time_mean": peeq_time_mean,
            # Legacy-compatible fields
            "EleLabelSet": ele_label_set,
            "EleTypeSet": ele_type_set,
            "EleMaterialIDSet": ele_material_id_set,
            "NodeConstrainedDof1": node_constrained_dof1,
            "NodeConstrainedDof2": node_constrained_dof2,
            "NodeConstrainedDof3": node_constrained_dof3,
            "EleNodeLabelSet": ele_node_label_set,
            "EleNodeCoordSet": ele_node_coord_set,
            "EleNodeDispSet": ele_node_disp_set,
            "EleGaussStressSet": ele_gauss_stress_set,
            "EleGaussStrainSet": ele_gauss_strain_set,
            "EleGaussPEEQSet": ele_gauss_peeq_set,
            "instance_name": np.asarray([str(inst.name)]),
            "step_name": np.asarray([str(final_step_name)]),
            "frame_index": np.asarray([[int(frame_index)]], dtype=np.int32),
            "frame_value": np.asarray([[float(final_frame_value)]], dtype=np.float64),
        }

        mat_path = _resolve_output_mat_path(odb_path, out_prefix=out_prefix, out_dir=out_dir)
        mat_dir = os.path.dirname(mat_path)
        if mat_dir and not os.path.isdir(mat_dir):
            os.makedirs(mat_dir)
        sio.savemat(mat_path, save_dict, do_compression=True)
        print("Saved unified MAT: {}".format(mat_path))
        return save_dict
    finally:
        odb.close()


def main():
    ap = argparse.ArgumentParser(
        description="Extract one unified MAT from ODB for both elastic comparison and legacy shakedown."
    )
    ap.add_argument("--odb", required=True, help="Path to Abaqus .odb")
    ap.add_argument("--out-prefix", default=None, help="Output prefix without suffix")
    ap.add_argument("--out-dir", default=None, help="Output directory; file name defaults to <odb_stem>.mat")
    ap.add_argument("--instance-name", default=None, help="ODB instance name")
    ap.add_argument("--step-name", default=None, help="ODB step name")
    ap.add_argument("--step", default=None, help="Alias of --step-name")
    ap.add_argument("--frame-index", type=int, default=-1, help="Legacy frame index inside selected step")
    ap.add_argument("--skip-initial", action="store_true", help="Skip frameValue==0 in elastic frame history")
    ap.add_argument("--fixed-dof1-set", default=None, help="Node set name for constrained DOF 1")
    ap.add_argument("--fixed-dof2-set", default=None, help="Node set name for constrained DOF 2")
    ap.add_argument("--fixed-dof3-set", default=None, help="Node set name for constrained DOF 3")
    ap.add_argument("--material-map", action="append", default=[], help="ELEMENT_SET:MATERIAL_ID")
    ap.add_argument("--default-material-id", type=int, default=1)
    args = ap.parse_args()

    export_odb(
        odb_path=args.odb,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir,
        instance_name=args.instance_name,
        step_name=args.step_name or args.step,
        frame_index=args.frame_index,
        skip_initial=args.skip_initial,
        fixed_dof1_set=args.fixed_dof1_set,
        fixed_dof2_set=args.fixed_dof2_set,
        fixed_dof3_set=args.fixed_dof3_set,
        material_map=_parse_material_map(args.material_map),
        default_material_id=args.default_material_id,
    )


if __name__ == "__main__":
    main()
