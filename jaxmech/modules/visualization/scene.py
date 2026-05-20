"""Single-scene rendering manager.

Wraps a PyVista off-screen plotter and provides an API for:

* loading ``VizData`` and building the grid
* switching the active field and component
* switching frames for multi-frame data
* adjusting colorbar range (auto / manual)
* preset camera views (+X, -X, +Y, -Y, +Z, -Z, iso)
* rotating / zooming via delta events
* returning JPEG frames for WebSocket streaming
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
except ImportError:
    pv = None  # type: ignore[assignment]

from jaxmech.modules.visualization.mat_reader import VizData, FieldInfo
from jaxmech.modules.visualization.mesh_builder import (
    build_grid,
    attach_field,
    scalarize_field,
    scalarize_per_gp,
)

# Preset camera positions: (position, focal_point, viewup)
_PRESET_VIEWS: dict[str, tuple] = {
    "+x": ((1, 0, 0), (0, 0, 0), (0, 0, 1)),
    "-x": ((-1, 0, 0), (0, 0, 0), (0, 0, 1)),
    "+y": ((0, 1, 0), (0, 0, 0), (0, 0, 1)),
    "-y": ((0, -1, 0), (0, 0, 0), (0, 0, 1)),
    "+z": ((0, 0, 1), (0, 0, 0), (0, 1, 0)),
    "-z": ((0, 0, -1), (0, 0, 0), (0, 1, 0)),
    "iso": ((1, 1, 1), (0, 0, 0), (0, 0, 1)),
}

_STRESS_COMP_6 = ["S11", "S22", "S33", "S12", "S13", "S23"]
_STRESS_COMP_3 = ["S11", "S22", "S12"]
_STRAIN_COMP_6 = ["E11", "E22", "E33", "E12", "E13", "E23"]
_STRAIN_COMP_3 = ["E11", "E22", "E12"]
_DISP_COMP = ["Ux", "Uy", "Uz"]
_FORCE_COMP = ["Fx", "Fy", "Fz"]
_FORCE_COMP_6 = ["F1", "F2", "F3", "M1", "M2", "M3"]
_SHELL_COMP_3 = ["11", "22", "12"]


def _is_rsdms_layer_stress(key_lower: str) -> bool:
    return key_lower.startswith(("rsdms_s_res", "rsdms_s_tot", "rsdm_s_res", "rsdm_s_tot"))


def _is_rsdms_layer_strain(key_lower: str) -> bool:
    return key_lower.startswith(("rsdms_e_res", "rsdms_e_tot", "rsdm_e_res", "rsdm_e_tot"))


def _is_rsdms_generalized(key_lower: str) -> bool:
    return key_lower.startswith(("rsdms_sf_", "rsdms_sm_", "rsdms_ge_", "rsdms_gk_"))


def _is_shell_generalized_subfield(key_lower: str) -> bool:
    return (
        _is_rsdms_generalized(key_lower)
        or key_lower.startswith(("shakedown_sf_", "shakedown_sm_", "shakedown_ge_", "shakedown_gk_"))
        or "generalized_residual_sf" in key_lower
        or "generalized_residual_sm" in key_lower
        or "generalized_total_sf" in key_lower
        or "generalized_total_sm" in key_lower
        or "generalized_residual_ge" in key_lower
        or "generalized_residual_gk" in key_lower
        or "generalized_total_ge" in key_lower
        or "generalized_total_gk" in key_lower
    )


def _is_shell_force_moment_vector(key_lower: str, text: str, n_comp: int) -> bool:
    return n_comp == 6 and (
        "shell generalized nodal force" in text
        or "force/moment" in text
        or key_lower in {
            "shell_gen_internal_force",
            "frame_nforc",
            "validation_jax_nforc",
            "validation_abaqus_nforc",
            "rsdms_nforc",
            "rsdms_ceq",
            "rsdms_ferror",
            "shakedown_equality_violation",
        }
    )


def _is_rsdm_excess(key_lower: str) -> bool:
    return key_lower.startswith(("rsdm_xs_", "rsdms_xs_"))


def _component_tag(fi: FieldInfo, comp: int) -> str:
    """Return a human-readable component tag for the colorbar title."""
    key_lower = fi.key.lower()
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    nc = fi.n_components
    is_stress = "stress" in text or _is_rsdms_layer_stress(key_lower)
    is_strain = "strain" in text or "peeq" in text or _is_rsdms_layer_strain(key_lower)
    is_disp = "displacement" in text or key_lower in {
        "u",
        "frame_u",
        "elastic_u",
        "solid_u_nodal",
        "shell_u_nodal",
        "validation_jax_u",
        "validation_abaqus_u",
    }
    is_force = "nforc" in text or "reaction" in text or "force" in text or key_lower in {
        "rsdms_ceq",
        "rsdms_ferror",
    }
    if "shell generalized stress" in text and nc == 6:
        labels = ["SF11", "SF22", "SF12", "SM11", "SM22", "SM12"]
        if 0 <= comp < len(labels):
            return labels[comp]
    if "shell generalized strain" in text and nc == 6:
        labels = ["GE11", "GE22", "GE12", "GK11", "GK22", "GK12"]
        if 0 <= comp < len(labels):
            return labels[comp]
    if "shell displacement" in text and nc == 6:
        labels = ["U1", "U2", "U3", "UR1", "UR2", "UR3"]
        if 0 <= comp < len(labels):
            return labels[comp]
    if _is_shell_force_moment_vector(key_lower, text, nc):
        labels = ["F1", "F2", "F3", "M1", "M2", "M3"]
        if 0 <= comp < len(labels):
            return labels[comp]
    for prefix, tokens in (
        ("SF", ("generalized_stress_sf", "generalized_residual_sf", "generalized_total_sf", "rsdms_sf_", "shakedown_sf_")),
        ("SM", ("generalized_stress_sm", "generalized_residual_sm", "generalized_total_sm", "rsdms_sm_", "shakedown_sm_")),
        ("GE", ("generalized_strain_ge", "rsdms_ge_")),
        ("GK", ("generalized_strain_gk", "rsdms_gk_")),
    ):
        if any(token in key_lower for token in tokens) and nc == 3 and 0 <= comp < len(_SHELL_COMP_3):
            return f"{prefix}{_SHELL_COMP_3[comp]}"
    if is_stress:
        labels = _STRESS_COMP_6 if nc == 6 else (_STRESS_COMP_3 if nc == 3 else None)
        if labels and 0 <= comp < len(labels):
            return labels[comp]
    elif is_strain:
        labels = _STRAIN_COMP_6 if nc == 6 else (_STRAIN_COMP_3 if nc == 3 else None)
        if labels and 0 <= comp < len(labels):
            return labels[comp]
    elif is_disp:
        labels = ["U1", "U2", "U3"] if key_lower.startswith("validation_") else _DISP_COMP
        if 0 <= comp < len(labels):
            return labels[comp]
    elif is_force:
        if key_lower.startswith("validation_"):
            labels = ["NFORC1", "NFORC2", "NFORC3"] if nc != 6 else ["F1", "F2", "F3", "M1", "M2", "M3"]
        else:
            labels = _FORCE_COMP_6 if nc == 6 else _FORCE_COMP
        if 0 <= comp < len(labels):
            return labels[comp]
    return f"comp {comp}"


def _default_scalar_tag(fi: Optional[FieldInfo]) -> str:
    if fi is None:
        return ""
    key_lower = fi.key.lower()
    if _is_rsdm_excess(key_lower):
        return "Mises"
    if fi.n_components <= 1:
        return ""
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    if _is_shell_generalized_subfield(key_lower):
        return "Magnitude"
    if "shell generalized stress" in text:
        return "SF magnitude"
    if "shell generalized strain" in text:
        return "GE magnitude"
    if "shell displacement" in text and fi.n_components == 6:
        return "U magnitude"
    if _is_shell_force_moment_vector(key_lower, text, fi.n_components):
        return "F magnitude"
    if "generalized_stress" in key_lower or "generalized_strain" in key_lower:
        return "Magnitude"
    if "stress" in text or _is_rsdms_layer_stress(key_lower) or _is_rsdm_excess(key_lower):
        return "Mises"
    if ("strain" in text or _is_rsdms_layer_strain(key_lower)) and "peeq" not in key_lower:
        return "Equivalent"
    if (
        key_lower in {"u", "frame_u", "elastic_u", "solid_u_nodal"}
        or "displacement" in text
        or "nforc" in text
        or "reaction" in text
        or "force" in text
        or key_lower in {"rsdms_ceq", "rsdms_ferror"}
    ):
        return "Magnitude"
    return ""


def _field_base_label(fi: Optional[FieldInfo]) -> str:
    if fi is None:
        return ""
    key_lower = fi.key.lower()
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    if "sgen_sf_magnitude" in key_lower:
        return "SGEN SF magnitude"
    if "sgen_sm_magnitude" in key_lower:
        return "SGEN SM magnitude"
    if "egen_ge_magnitude" in key_lower:
        return "EGEN GE magnitude"
    if "egen_gk_magnitude" in key_lower:
        return "EGEN GK magnitude"
    if "u_trans_magnitude" in key_lower:
        return "U magnitude"
    if "u_rot_magnitude" in key_lower:
        return "UR magnitude"
    if "nforc_force_magnitude" in key_lower:
        return "NFORC force magnitude"
    if "nforc_moment_magnitude" in key_lower:
        return "NFORC moment magnitude"
    if "ceq_force_magnitude" in key_lower:
        return "CEQ force magnitude"
    if "ceq_moment_magnitude" in key_lower:
        return "CEQ moment magnitude"
    if "ferror_force_magnitude" in key_lower:
        return "FERROR force magnitude"
    if "ferror_moment_magnitude" in key_lower:
        return "FERROR moment magnitude"
    if "peeq" in text:
        return "PEEQ"
    if "shell generalized stress" in text:
        return "Shell generalized stress"
    if "shell generalized strain" in text:
        return "Shell generalized strain"
    if _is_rsdm_excess(key_lower):
        return "Excess stress"
    if "stress" in text or _is_rsdms_layer_stress(key_lower):
        return "Stress"
    if "strain" in text or _is_rsdms_layer_strain(key_lower):
        return "Strain"
    if key_lower.startswith(("rsdms_sf_", "rsdms_sm_", "shakedown_sf_", "shakedown_sm_")) or "generalized_residual_sf" in key_lower or "generalized_total_sf" in key_lower or "generalized_residual_sm" in key_lower or "generalized_total_sm" in key_lower:
        return "Gen. stress"
    if key_lower.startswith(("rsdms_ge_", "rsdms_gk_")):
        return "Gen. strain"
    if key_lower in {"u", "frame_u", "elastic_u", "solid_u_nodal", "validation_jax_u", "validation_abaqus_u"} or "displacement" in text:
        return "Displacement"
    if "force_error" in key_lower or key_lower == "rsdms_ferror":
        return "Force error"
    if "nforc" in text or "reaction" in text or "force" in text or key_lower == "rsdms_ceq":
        return "NFORC"
    return fi.label or fi.key


def _validation_side_label(key: str) -> str:
    key_lower = str(key or "").lower()
    if key_lower.startswith("validation_jax_"):
        return "JAX"
    if key_lower.startswith("validation_abaqus_"):
        return "ABAQUS"
    return ""


DEFAULT_WINDOW_SIZE = (800, 600)
STREAM_QUALITY = 85  # JPEG quality for WebSocket frames
_UNSET = object()


def _preferred_initial_field(vd: VizData) -> str | None:
    for key in (
        "validation_jax_sgen_sf_magnitude",
        "validation_abaqus_sgen_sf_magnitude",
        "frame_shell_sgen_sf_magnitude",
        "gauss_shell_sgen_sf_magnitude",
        "shell_sgen_sf_magnitude",
        "derived_vm_frame_gauss_stress",
        "derived_vm_gauss_stress",
    ):
        if key in vd.field_info:
            return key
    keys = vd.field_keys()
    return keys[0] if keys else None


class VizScene:
    """Manages one PyVista off-screen plotter bound to a ``VizData``."""

    def __init__(self, window_size: tuple[int, int] = DEFAULT_WINDOW_SIZE):
        if pv is None:
            raise RuntimeError("PyVista is required: pip install pyvista")
        self._window_size = window_size
        self._plotter: Optional[pv.Plotter] = None
        self._vd: Optional[VizData] = None
        self._grid: Optional[pv.UnstructuredGrid] = None
        self._base_points: Optional[np.ndarray] = None  # undeformed coords
        self._active_field: Optional[str] = None
        self._active_frame: int = 0
        self._active_component: Optional[int] = None
        self._clim: Optional[tuple[float, float]] = None
        self._cmap: str = "coolwarm"
        # Base font size (pt) for the scalar-bar label / scalar-bar title.
        # The actual rendered size still goes through ``_font_scale`` so it
        # auto-scales with window size; this multiplier lets users bump
        # the legend text up/down for large-screen presentations.
        self._legend_font_pt: int = 10
        self._show_edges: bool = False
        self._deform_scale: float = 0.0  # 0 = off
        self._show_overlay: bool = False
        self._clip_enabled: bool = False
        self._clip_axis: str = "z"
        self._clip_position: float = 0.5  # normalized 0..1 along axis bounds
        self._clip_invert: bool = False
        self._main_actor = None
        self._title_actor = None
        # Reverse index: original-mesh node_id -> list of cell_ids that
        # share that node. Built once on ``load()`` so the probe can
        # answer "all elements that own this node" in O(1).
        self._node_to_cells: Optional[Dict[int, List[int]]] = None
        self._selected_cell_ids: set[int] = set()
        self._selection_undo_stack: list[list[int]] = []
        self._hidden_cell_ids: set[int] = set()
        self._hidden_undo_stack: list[list[int]] = []
        self._visible_grid_cache = None
        self._visible_grid_cache_key: Optional[tuple[int, ...]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, vd: VizData) -> list[str]:
        """Load visualisation data and return available field keys."""
        self._vd = vd
        self._grid = build_grid(vd)
        self._base_points = np.array(self._grid.points, copy=True)
        # Tag the master grid with identity arrays so ``clip()`` will
        # scatter them to every surviving cell / point. The probe then
        # reads these back from ``_display_grid()`` to recover the
        # ORIGINAL mesh ids, even when the user is interacting with a
        # cutting plane.
        try:
            self._grid.point_data["_orig_point_id"] = np.arange(
                self._grid.n_points, dtype=np.int32
            )
            self._grid.cell_data["_orig_cell_id"] = np.arange(
                self._grid.n_cells, dtype=np.int32
            )
        except Exception:
            pass
        self._active_field = None
        self._active_frame = 0
        self._clim = None
        self._deform_scale = 0.0
        self._clip_enabled = False
        self._clip_axis = "z"
        self._clip_position = 0.5
        self._clip_invert = False
        self._main_actor = None
        self._title_actor = None
        self._node_to_cells = self._build_node_to_cells()
        self._selected_cell_ids = set()
        self._selection_undo_stack = []
        self._hidden_cell_ids = set()
        self._hidden_undo_stack = []
        self._visible_grid_cache = None
        self._visible_grid_cache_key = None
        self._rebuild_plotter()

        keys = vd.field_keys()
        initial_field = _preferred_initial_field(vd)
        if initial_field:
            self.set_field(initial_field)
        self.set_preset_view("iso")
        return keys

    # ------------------------------------------------------------------
    # Field / frame switching
    # ------------------------------------------------------------------

    def set_field(
        self,
        key: str,
        *,
        component: Optional[int] = None,
        frame: Optional[int] = None,
    ) -> None:
        """Switch the active field variable."""
        if self._vd is None or self._grid is None:
            return
        fi = self._vd.field_info.get(key)
        if fi is None:
            return

        self._active_field = key
        self._active_component = component
        if frame is not None:
            self._active_frame = frame

        self._apply_field()

    def set_frame(self, frame: int) -> None:
        """Switch the active time frame (for multi-frame fields)."""
        self._active_frame = max(0, frame)
        if self._active_field is not None:
            self._apply_field()

    def set_component(self, component: Optional[int]) -> None:
        """Switch which stress/strain component to display."""
        self._active_component = component
        if self._active_field is not None:
            self._apply_field()

    def _apply_field(self) -> None:
        """Internal: update the grid's scalars and refresh the plotter."""
        _, fi, frame_data = self._resolve_field_data()

        attach_field(
            self._grid, frame_data, "active",
            self._vd, location=fi.location,
            component=self._active_component,
            field_key=fi.key,
        )
        if self._deform_scale != 0.0:
            self._apply_deformation()
        self._rebuild_plotter()

    # ------------------------------------------------------------------
    # Colorbar / legend
    # ------------------------------------------------------------------

    def set_clim(self, vmin: Optional[float], vmax: Optional[float]) -> None:
        """Set colorbar limits. Pass ``None`` for auto range."""
        if vmin is not None and vmax is not None:
            self._clim = (float(vmin), float(vmax))
        else:
            self._clim = None
        self._rebuild_plotter()

    def get_auto_clim(self) -> Optional[tuple[float, float]]:
        """Return the data-range (min, max) of the active scalar field."""
        if self._grid is None or "active" not in self._grid.point_data:
            return None
        arr = self._grid.point_data["active"]
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        vmin = float(finite.min())
        vmax = float(finite.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmin == vmax:
            pad = max(abs(vmin) * 1.0e-6, 1.0e-12)
            return (vmin - pad, vmax + pad)
        return (vmin, vmax)

    def set_colormap(self, cmap: str) -> None:
        self._cmap = cmap

    def set_legend_font_pt(self, pt: int) -> None:
        """Set base font size (pt) for the colour-bar numeric labels only.

        Affects only ``scalar_bar_args["label_font_size"]`` — the tick
        labels like ``1.00e-12`` on the colour bar. The variable-name
        title at the top-left of the canvas keeps its fixed baseline so
        the user can tweak the legend's number font without inflating
        the title text. Allowed range is clipped to [6, 28]. The change
        requires a plotter rebuild because PyVista bakes
        ``scalar_bar_args`` into the actor at ``add_mesh`` time.
        """
        try:
            value = int(pt)
        except (TypeError, ValueError):
            return
        value = max(6, min(28, value))
        if value == self._legend_font_pt:
            return
        self._legend_font_pt = value
        self._rebuild_plotter()

    def set_show_edges(self, show: bool) -> None:
        self._show_edges = show
        self._rebuild_plotter()

    def set_show_overlay(self, show: bool) -> None:
        self._show_overlay = show
        self._rebuild_plotter()

    # ------------------------------------------------------------------
    # Deformed shape
    # ------------------------------------------------------------------

    def set_deform_scale(self, scale: float) -> None:
        """Set the displacement magnification factor (0 = undeformed)."""
        self._deform_scale = float(scale)
        self._apply_deformation()
        self._rebuild_plotter()

    # ------------------------------------------------------------------
    # Section clip (cutting plane)
    # ------------------------------------------------------------------

    _CLIP_AXIS_INDEX = {"x": 0, "y": 2, "z": 4}
    _CLIP_AXIS_NORMAL = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }

    def set_clip(
        self,
        *,
        enabled: Optional[bool] = None,
        axis: Optional[str] = None,
        position: Optional[float] = None,
        invert: Optional[bool] = None,
    ) -> None:
        """Enable or update an axis-aligned cutting plane through the mesh.

        Performance: when the plotter already exists we only replace the
        main mesh actor in place (no plotter recreation), which keeps the
        UI responsive while dragging the position slider.
        """
        if enabled is not None:
            self._clip_enabled = bool(enabled)
        if axis is not None:
            axis_key = str(axis).lower()
            if axis_key in self._CLIP_AXIS_INDEX:
                self._clip_axis = axis_key
        if position is not None:
            self._clip_position = float(np.clip(float(position), 0.0, 1.0))
        if invert is not None:
            self._clip_invert = bool(invert)
        if not self._refresh_main_actor_in_place():
            self._rebuild_plotter()

    def _clip_bounds_info(self) -> dict[str, list[float]]:
        if self._grid is None:
            return {}
        bounds = self._grid.bounds
        return {
            "x": [float(bounds[0]), float(bounds[1])],
            "y": [float(bounds[2]), float(bounds[3])],
            "z": [float(bounds[4]), float(bounds[5])],
        }

    def _clip_world_coordinate(self) -> Optional[float]:
        """Return the cutting-plane coordinate along the active axis.

        A tiny extension is added beyond the bounds so that the slider
        extremes (0 and 1) reliably correspond to "no cut" and "fully
        cut" without floating-point edge cases removing border cells.
        """
        bounds = self._clip_bounds_info()
        axis = self._clip_axis
        span = bounds.get(axis)
        if not span:
            return None
        lo, hi = float(span[0]), float(span[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return lo
        eps = (hi - lo) * 1.0e-4
        return (lo - eps) + float(self._clip_position) * (hi - lo + 2.0 * eps)

    def _clip_origin(self) -> Optional[tuple[float, float, float]]:
        if self._grid is None:
            return None
        bounds = self._grid.bounds
        cx = 0.5 * (float(bounds[0]) + float(bounds[1]))
        cy = 0.5 * (float(bounds[2]) + float(bounds[3]))
        cz = 0.5 * (float(bounds[4]) + float(bounds[5]))
        coord = self._clip_world_coordinate()
        if coord is None:
            return None
        if self._clip_axis == "x":
            return (coord, cy, cz)
        if self._clip_axis == "y":
            return (cx, coord, cz)
        return (cx, cy, coord)

    def _display_grid(self) -> Optional["pv.UnstructuredGrid"]:
        """Return the mesh actually drawn (optionally clipped).

        Behaviour at the slider extremes (ABAQUS-style):

        * position 0 (no extra cut) returns the full mesh.
        * position 1 (cut everything away) returns ``None`` so the
          rendered scene is empty rather than silently snapping back
          to the full mesh.
        """
        base_grid = self._visible_base_grid()
        if base_grid is None:
            return None
        if not self._clip_enabled:
            return base_grid
        origin = self._clip_origin()
        normal = self._CLIP_AXIS_NORMAL.get(self._clip_axis)
        if origin is None or normal is None:
            return base_grid
        try:
            clipped = base_grid.clip(
                normal=normal,
                origin=origin,
                invert=bool(self._clip_invert),
            )
            if clipped is None or int(getattr(clipped, "n_cells", 0)) <= 0:
                return None
            return clipped
        except Exception:
            return base_grid

    def _visible_base_grid(self) -> Optional["pv.UnstructuredGrid"]:
        """Return the original grid after session-level hidden elements."""
        if self._grid is None:
            return None
        if not self._hidden_cell_ids:
            return self._grid
        key = tuple(sorted(int(c) for c in self._hidden_cell_ids))
        if self._visible_grid_cache is not None and self._visible_grid_cache_key == key:
            return self._visible_grid_cache
        try:
            n_cells = int(self._grid.n_cells)
            hidden = {c for c in key if 0 <= c < n_cells}
            keep = np.asarray([idx for idx in range(n_cells) if idx not in hidden], dtype=np.int64)
            if keep.size == 0:
                self._visible_grid_cache = None
                self._visible_grid_cache_key = key
                return None
            visible = self._grid.extract_cells(keep)
            self._visible_grid_cache = visible
            self._visible_grid_cache_key = key
            return visible
        except Exception:
            return self._grid

    def _resolve_field_data(
        self,
        key: Optional[str] = None,
        frame: Optional[int] = None,
    ) -> tuple[str, FieldInfo, np.ndarray]:
        """Return the resolved field key, metadata, and single-frame data."""
        if self._vd is None:
            raise RuntimeError("No visualization data loaded.")
        resolved_key = key or self._active_field
        if not resolved_key:
            raise RuntimeError("No active field selected.")
        fi = self._vd.field_info.get(resolved_key)
        if fi is None:
            raise RuntimeError(f"Unknown field: {resolved_key}")
        raw = self._vd.fields[resolved_key]
        if fi.is_multiframe:
            frame_idx = self._active_frame if frame is None else int(frame)
            frame_idx = max(0, min(frame_idx, fi.n_frames - 1))
            frame_data = raw[frame_idx]
        elif (
            isinstance(raw, np.ndarray)
            and raw.ndim == 2
            and raw.shape[0] == 1
            and fi.n_components == 1
            and fi.location in {"gauss", "node", "element"}
        ):
            frame_data = raw[0]
        elif (
            isinstance(raw, np.ndarray)
            and raw.ndim >= 3
            and raw.shape[0] == 1
            and fi.location in {"gauss", "node", "element"}
        ):
            frame_data = raw[0]
        else:
            frame_data = raw
        return resolved_key, fi, frame_data

    def get_scalar_data(
        self,
        key: Optional[str] = None,
        *,
        component: object = _UNSET,
        frame: Optional[int] = None,
    ) -> np.ndarray:
        """Return the scalar nodal values currently used for colouring."""
        _, fi, frame_data = self._resolve_field_data(key=key, frame=frame)
        resolved_component = self._active_component if component is _UNSET else component
        return scalarize_field(
            frame_data,
            self._vd,
            location=fi.location,
            component=resolved_component,
            field_key=fi.key,
        )

    def get_scalar_frame_series(self) -> Optional[np.ndarray]:
        """Return scalar nodal values for every frame of the active field."""
        if self._vd is None or not self._active_field:
            return None
        fi = self._vd.field_info.get(self._active_field)
        if fi is None or not fi.is_multiframe:
            return None
        return np.stack([self.get_scalar_data(frame=i) for i in range(fi.n_frames)], axis=0)

    def active_field_info(self) -> Optional[FieldInfo]:
        """Return metadata for the currently active field."""
        if self._vd is None or not self._active_field:
            return None
        return self._vd.field_info.get(self._active_field)

    def register_scalar_field(
        self,
        key: str,
        values: np.ndarray,
        *,
        label: str = "Diff",
        n_frames: int = 1,
        metadata: Optional[dict] = None,
        set_active: bool = False,
        frame: Optional[int] = None,
    ) -> None:
        """Register a generated nodal scalar field in the current scene."""
        if self._vd is None:
            raise RuntimeError("No visualization data loaded.")
        arr = np.asarray(values, dtype=np.float64)
        n_frames = max(1, int(n_frames))
        if n_frames <= 1:
            arr = arr.reshape(-1)
        elif arr.ndim != 2 or arr.shape[0] != n_frames:
            raise ValueError(f"Expected scalar frame field with shape ({n_frames}, n_values).")
        self._vd.fields[key] = arr
        self._vd.field_info[key] = FieldInfo(
            key=key,
            label=label,
            n_components=1,
            location="node",
            n_frames=n_frames,
        )
        if metadata is not None:
            generated = self._vd.metadata.setdefault("generated_field_info", {})
            generated[key] = dict(metadata)
        if set_active:
            target_frame = 0 if n_frames <= 1 else max(0, min(int(frame or 0), n_frames - 1))
            self._clim = None
            self.set_field(key, component=None, frame=target_frame)
        elif self._active_field == key:
            self._apply_field()

    def _apply_deformation(self) -> None:
        """Warp grid points by the active displacement field."""
        if self._grid is None or self._base_points is None or self._vd is None:
            return
        if self._deform_scale == 0.0:
            self._grid.points = np.array(self._base_points, copy=True)
            return

        # Find the displacement field for the current frame
        disp = self._find_displacement_array()
        if disp is None:
            self._grid.points = np.array(self._base_points, copy=True)
            return

        n_nodes = self._base_points.shape[0]
        n_dim = self._base_points.shape[1]
        arr = np.asarray(disp, dtype=np.float64)
        if arr.ndim == 1:
            ndof = arr.size // n_nodes if n_nodes > 0 else n_dim
            arr = arr.reshape(n_nodes, ndof)
        if arr.shape[1] > n_dim:
            arr = arr[:, :n_dim]
        elif arr.shape[1] < n_dim:
            arr = np.hstack([arr, np.zeros((n_nodes, n_dim - arr.shape[1]))])

        self._grid.points = self._base_points + arr * self._deform_scale

    def _find_displacement_array(self) -> Optional[np.ndarray]:
        """Locate the displacement array for the current frame."""
        vd = self._vd
        frame = self._active_frame

        # Multi-frame displacement
        if "frame_u" in vd.fields:
            raw = vd.fields["frame_u"]
            fi = vd.field_info["frame_u"]
            idx = min(frame, fi.n_frames - 1)
            return raw[idx]

        # Single-frame
        if "u" in vd.fields:
            return vd.fields["u"]

        if "elastic_u" in vd.fields:
            return vd.fields["elastic_u"]

        if "solid_u_nodal" in vd.fields:
            return vd.fields["solid_u_nodal"]

        return None

    # ------------------------------------------------------------------
    # Camera / view
    # ------------------------------------------------------------------

    def set_preset_view(self, name: str) -> None:
        """Apply a named preset view ('+x', '-x', '+y', '-y', '+z', '-z', 'iso')."""
        preset = _PRESET_VIEWS.get(name.lower())
        if preset is None or self._plotter is None:
            return
        pos, focal, up = preset
        bounds = self._grid.bounds if self._grid is not None else (-1, 1, -1, 1, -1, 1)
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        diag = np.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )
        dist = diag * 2.0
        cam_pos = (
            cx + pos[0] * dist,
            cy + pos[1] * dist,
            cz + pos[2] * dist,
        )
        self._plotter.camera_position = [cam_pos, (cx, cy, cz), up]
        self._fit_camera_to_grid()
        self._plotter.reset_camera_clipping_range()
        self._plotter.camera.Modified()

    def orbit(self, dx: float, dy: float) -> None:
        """Rotate the camera by pixel deltas (from mouse drag)."""
        if self._plotter is None:
            return
        azimuth = -dx * 0.5
        elevation = -dy * 0.5
        self._plotter.camera.Azimuth(azimuth)
        self._plotter.camera.Elevation(elevation)
        self._plotter.camera.OrthogonalizeViewUp()
        self._plotter.reset_camera_clipping_range()
        self._plotter.camera.Modified()

    def pan(self, dx: float, dy: float) -> None:
        """Pan the camera by pixel deltas."""
        if self._plotter is None:
            return
        cam = self._plotter.camera
        focal = np.array(cam.focal_point)
        position = np.array(cam.position)
        view_up = np.array(cam.up)
        view_dir = focal - position
        dist = np.linalg.norm(view_dir)
        scale = dist * 0.001
        right = np.cross(view_dir, view_up)
        right = right / (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, view_dir)
        up = up / (np.linalg.norm(up) + 1e-12)
        shift = right * (-dx * scale) + up * (dy * scale)
        cam.focal_point = tuple(focal + shift)
        cam.position = tuple(position + shift)
        self._plotter.reset_camera_clipping_range()
        cam.Modified()

    def zoom(self, factor: float) -> None:
        """Zoom the camera by a factor (>1 = zoom in)."""
        if self._plotter is None:
            return
        self._plotter.camera.Zoom(factor)
        self._plotter.reset_camera_clipping_range()
        self._plotter.camera.Modified()

    def get_camera_state(self) -> Optional[dict]:
        """Return serialisable camera state."""
        if self._plotter is None:
            return None
        cam = self._plotter.camera
        return {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up": list(cam.up),
            "view_angle": float(getattr(cam, "view_angle", 30.0)),
            "parallel_scale": float(getattr(cam, "parallel_scale", 1.0)),
            "parallel_projection": bool(getattr(cam, "parallel_projection", False)),
            "clipping_range": list(getattr(cam, "clipping_range", (0.0, 1.0))),
        }

    def set_camera_state(self, state: dict) -> None:
        """Restore camera from a serialised state dict."""
        if self._plotter is None or state is None:
            return
        self._plotter.camera_position = [
            tuple(state["position"]),
            tuple(state["focal_point"]),
            tuple(state["up"]),
        ]
        cam = self._plotter.camera
        if "parallel_projection" in state and hasattr(cam, "parallel_projection"):
            cam.parallel_projection = bool(state["parallel_projection"])
        if "parallel_scale" in state and hasattr(cam, "parallel_scale"):
            cam.parallel_scale = float(state["parallel_scale"])
        if "view_angle" in state and hasattr(cam, "view_angle"):
            cam.view_angle = float(state["view_angle"])
        if "clipping_range" in state and hasattr(cam, "clipping_range"):
            try:
                cam.clipping_range = tuple(float(v) for v in state["clipping_range"])
            except Exception:
                pass
        self._plotter.reset_camera_clipping_range()
        self._plotter.camera.Modified()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_jpeg(self, quality: int = STREAM_QUALITY) -> bytes:
        """Render current scene and return JPEG bytes."""
        if self._plotter is None:
            return b""
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True)
        from PIL import Image
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def render_png(self) -> bytes:
        """Render current scene and return lossless PNG bytes."""
        if self._plotter is None:
            return b""
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True)
        from PIL import Image
        pil = Image.fromarray(img)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()

    @property
    def info(self) -> dict:
        """Current scene state for the frontend."""
        fi = self._vd.field_info.get(self._active_field, None) if self._vd else None
        has_disp = False
        if self._vd:
            has_disp = any(
                k in self._vd.fields
                for k in ("u", "frame_u", "elastic_u", "solid_u_nodal")
            )
        return {
            "active_field": self._active_field,
            "active_field_label": fi.label if fi else "",
            "active_scalar_label": self._field_label(),
            "active_frame": self._active_frame,
            "active_component": self._active_component,
            "n_frames": fi.n_frames if fi else 0,
            "n_components": fi.n_components if fi else 0,
            "clim": list(self._clim) if self._clim else None,
            "auto_clim": list(self.get_auto_clim()) if self.get_auto_clim() else None,
            "cmap": self._cmap,
            "legend_font_pt": self._legend_font_pt,
            "show_edges": self._show_edges,
            "show_overlay": self._show_overlay,
            "deform_scale": self._deform_scale,
            "has_displacement": has_disp,
            "clip": {
                "enabled": bool(self._clip_enabled),
                "axis": self._clip_axis,
                "position": float(self._clip_position),
                "invert": bool(self._clip_invert),
                "coordinate": self._clip_world_coordinate(),
                "bounds": self._clip_bounds_info(),
            },
            "element_selection": {
                "selected_count": len(self._selected_cell_ids),
                "can_selection_undo": bool(self._selection_undo_stack),
                "hidden_count": len(self._hidden_cell_ids),
                "can_undo": bool(self._hidden_undo_stack),
            },
            "fields": list(self._vd.field_info.keys()) if self._vd else [],
            "field_labels": {
                k: v.label for k, v in self._vd.field_info.items()
            } if self._vd else {},
            "field_meta": {
                k: {
                    "label": v.label,
                    "n_components": v.n_components,
                    "location": v.location,
                    "n_frames": v.n_frames,
                    "symbol": v.symbol,
                    "formula": v.formula,
                    "description": v.description,
                }
                for k, v in self._vd.field_info.items()
            } if self._vd else {},
            "frame_times": self._vd.frame_times.tolist() if (
                self._vd and self._vd.frame_times is not None
            ) else None,
            "frame_labels": self._frame_labels(fi),
            "frame_details": self._frame_details(fi),
            "generated_field_info": self._vd.metadata.get("generated_field_info", {}) if self._vd else {},
            "source": self._vd.source_path if self._vd else "",
            # Stable MAT identity for probe CSV export (source path +
            # file mtime + size + manifest producer / analysis type).
            # Kept in ``info`` so the frontend gets it for free on every
            # ``send_info`` push, without an extra round-trip.
            "mat_identity": self._mat_identity(),
        }

    # ------------------------------------------------------------------
    # Internal plotter management
    # ------------------------------------------------------------------

    def _rebuild_plotter(self) -> None:
        """Recreate the plotter with current settings."""
        if self._plotter is not None:
            cam_state = self.get_camera_state()
            try:
                self._plotter.close()
            except Exception:
                pass
        else:
            cam_state = None

        self._plotter = pv.Plotter(
            off_screen=True,
            window_size=self._window_size,
        )
        self._plotter.set_background("white")
        self._main_actor = None
        self._title_actor = None

        self._add_main_mesh_actor()

        if self._show_overlay:
            self._add_overlay()

        if cam_state is not None:
            self.set_camera_state(cam_state)
        else:
            self.set_preset_view("iso")

    def _build_main_mesh_kwargs(self) -> dict:
        """Common ``add_mesh`` kwargs for the main scalar mesh."""
        font_scale = self._font_scale()
        kwargs: dict = {
            "scalars": "active",
            "cmap": self._cmap,
            "show_edges": self._show_edges,
            "scalar_bar_args": {
                "title": "",
                "color": "black",
                # ``title_font_size`` only matters if we ever put a title
                # back into the bar — kept at a fixed baseline so the
                # legend-font dropdown affects only the numeric tick
                # labels, not the colour-bar title.
                "title_font_size": int(round(11 * font_scale)),
                # ``label_font_size`` controls the numeric labels on the
                # colour bar (e.g. ``1.00e-12``). This is the only thing
                # the user-facing 图例字体 dropdown should affect.
                "label_font_size": int(round(self._legend_font_pt * font_scale)),
                "vertical": True,
                "position_x": 0.03,
                "position_y": 0.22,
                "width": 0.09,
                "height": 0.58,
            },
        }
        if self._clim is not None:
            kwargs["clim"] = self._clim
        else:
            auto_clim = self.get_auto_clim()
            if auto_clim is not None:
                kwargs["clim"] = auto_clim
        return kwargs

    def _add_main_mesh_actor(self) -> None:
        """Add the main mesh actor (replacing any existing one by name)."""
        if self._plotter is None:
            return
        display_grid = self._display_grid()
        if display_grid is None:
            # Fully clipped or empty: drop the previous actor so the
            # scene goes blank instead of stale-rendering.
            if self._main_actor is not None:
                try:
                    self._plotter.remove_actor(self._main_actor)
                except Exception:
                    pass
                self._main_actor = None
            # Also try removing by name in case the reference is stale.
            try:
                self._plotter.remove_actor("main_mesh")
            except Exception:
                pass
            return
        if "active" in display_grid.point_data:
            kwargs = self._build_main_mesh_kwargs()
            self._main_actor = self._plotter.add_mesh(
                display_grid, name="main_mesh", **kwargs
            )
            self._title_actor = self._add_scalar_bar_title()
        else:
            self._main_actor = self._plotter.add_mesh(
                display_grid,
                name="main_mesh",
                color="lightgray",
                show_edges=True,
            )

    def _refresh_main_actor_in_place(self) -> bool:
        """Replace the main mesh actor without recreating the plotter.

        This is the fast path used by interactive controls (cutting
        plane slider, etc.). It is roughly an order of magnitude faster
        than ``_rebuild_plotter`` because it skips VTK render-window
        teardown and camera restoration.
        """
        if self._plotter is None:
            return False
        try:
            if self._title_actor is not None:
                try:
                    self._plotter.remove_actor(self._title_actor)
                except Exception:
                    pass
                self._title_actor = None
            self._add_main_mesh_actor()
            return True
        except Exception:
            return False

    def _font_scale(self) -> float:
        width, height = self._window_size
        return max(1.0, min(width / DEFAULT_WINDOW_SIZE[0], height / DEFAULT_WINDOW_SIZE[1]))

    def _fit_camera_to_grid(self, margin: float = 1.08, zoom: float = 1.35) -> None:
        """Fit the current camera orientation to the grid with a small margin."""
        grid = self._display_grid() or self._grid
        if self._plotter is None or grid is None:
            return
        try:
            self._plotter.reset_camera(bounds=grid.bounds)
        except TypeError:
            self._plotter.reset_camera()
        if zoom > 0:
            self._plotter.camera.Zoom(float(zoom))
        if margin > 1.0:
            cam = self._plotter.camera
            if bool(getattr(cam, "parallel_projection", False)):
                cam.parallel_scale = float(getattr(cam, "parallel_scale", 1.0)) * float(margin)
            else:
                cam.Zoom(1.0 / float(margin))

    def _add_scalar_bar_title(self):
        """Place the scalar title in the top-left corner of the canvas.

        PyVista ``add_text`` uses a bottom-left origin in viewport pixels,
        so ``y = height - margin - font_height`` keeps the text glued to
        the very top of the panel. This guarantees the legend caption
        (e.g. "JAX SGEN SF magnitude") stays well above the model and
        never overlaps it in screenshots.

        When the "信息" overlay is on, ``_add_overlay`` already renders
        the same field label at the upper-left corner; skip the scalar
        title in that case so the two texts do not stack on top of each
        other.

        Returns the created text actor (or ``None``) so callers can later
        remove it during in-place refreshes.
        """
        if self._plotter is None:
            return None
        if self._show_overlay:
            return None
        title = self._title_label(compact=True)
        if not title:
            return None
        width, height = self._window_size
        # The variable-name title (e.g. "JAX SGEN SF magnitude") is
        # rendered as a separate ``add_text`` actor in the top-left of
        # the canvas. Its font is deliberately independent of the
        # ``_legend_font_pt`` knob — that dropdown is meant for the
        # numeric tick labels on the colour bar only.
        font_size = self._top_title_font_size(title, max(1, width - 24))
        margin_top_px = 6
        # Approximate glyph height ≈ 1.5×font_size in pixels.
        y = max(0, height - margin_top_px - int(round(font_size * 1.5)))
        x = 6
        try:
            return self._plotter.add_text(
                title,
                position=(x, y),
                font_size=font_size,
                color="black",
                shadow=False,
            )
        except Exception:
            return None

    def _field_label(self) -> str:
        if self._active_field is None or self._vd is None:
            return ""
        fi = self._vd.field_info.get(self._active_field)
        if fi is None:
            return self._active_field
        label = _field_base_label(fi)
        side_label = _validation_side_label(fi.key)
        if side_label and not label.lower().startswith(side_label.lower()):
            label = f"{side_label} {label}"
        if self._active_component is not None:
            comp = self._active_component
            tag = _component_tag(fi, comp)
            label += f" - {tag}"
        else:
            default_tag = _default_scalar_tag(fi)
            if default_tag:
                label += f" - {default_tag}"
        return label

    def _title_label(self, *, compact: bool = False) -> str:
        field = self._field_label()
        if not field or self._vd is None or self._active_field is None:
            return field
        fi = self._vd.field_info.get(self._active_field)
        frame = self._compact_frame_label(fi) if compact else self._frame_label(fi)
        return f"{field}  {frame}" if frame else field

    def _compact_frame_label(self, fi: Optional[FieldInfo]) -> str:
        label = self._frame_label(fi)
        if not label:
            return ""
        frame_prefix = ""
        if " | " in label:
            frame_prefix, detail = label.split(" | ", 1)
            frame_prefix = frame_prefix.strip()
            path_match = re.search(r"(Path\s+\d+/\d+:\s+load\s+.+)$", detail)
            if path_match:
                path_text = path_match.group(1)
                path_text = path_text.replace("Path ", "Path ")
                path_text = path_text.replace(": load ", ": ")
                path_text = path_text.replace(" -> ", "->")
                return f"{frame_prefix}  {path_text}".strip()
            if frame_prefix == "Frame 1/1":
                label = detail
            else:
                label = f"{frame_prefix}  {detail.strip()}"
        label = label.replace("alpha =", "alpha=")
        label = label.replace("; load = ", "; load=")
        label = re.sub(r"\s+", " ", label).strip()
        return label

    def _frame_label(self, fi: Optional[FieldInfo]) -> str:
        if fi is None:
            return ""
        total_frames = max(int(fi.n_frames), 1)
        active_frame = max(0, min(self._active_frame, total_frames - 1)) + 1
        labels = self._frame_labels(fi)
        if labels and 0 <= active_frame - 1 < len(labels):
            return f"Frame {active_frame}/{total_frames} | {labels[active_frame - 1]}"
        label = f"Frame {active_frame}/{total_frames}"
        if total_frames <= 1 or self._vd is None or self._vd.frame_times is None:
            return label
        frame_times = np.asarray(self._vd.frame_times, dtype=np.float64).reshape(-1)
        if frame_times.size == 0:
            return label
        idx = min(active_frame - 1, frame_times.size - 1)
        current_time = float(frame_times[idx])
        total_time = float(frame_times[-1])
        if np.isfinite(current_time) and np.isfinite(total_time):
            label += f"  time={current_time:g}/{total_time:g}"
        return label

    def _frame_labels(self, fi: Optional[FieldInfo]) -> list[str]:
        if fi is None or self._vd is None:
            return []
        by_field = self._vd.metadata.get("shakedown_frame_labels_by_field", {})
        if isinstance(by_field, dict):
            labels = by_field.get(str(fi.key), [])
            if isinstance(labels, np.ndarray):
                labels = labels.reshape(-1).tolist()
            if isinstance(labels, (list, tuple)):
                out = [str(item) for item in labels]
                if len(out) >= int(fi.n_frames):
                    return out[: int(fi.n_frames)]
        if str(fi.key) in {"shakedown_total_stress", "shakedown_inequality_violation"}:
            labels = self._vd.metadata.get("shakedown_vertex_labels", [])
            if isinstance(labels, np.ndarray):
                labels = labels.reshape(-1).tolist()
            if isinstance(labels, (list, tuple)):
                out = [str(item) for item in labels]
                if len(out) >= int(fi.n_frames):
                    return out[: int(fi.n_frames)]
        return []

    def _frame_details(self, fi: Optional[FieldInfo]) -> list[dict]:
        if fi is None or self._vd is None:
            return []
        by_field = self._vd.metadata.get("shakedown_frame_details_by_field", {})
        if not isinstance(by_field, dict):
            return []
        details = by_field.get(str(fi.key), [])
        if not isinstance(details, (list, tuple)):
            return []
        out = [dict(item) for item in details if isinstance(item, dict)]
        if len(out) >= int(fi.n_frames):
            return out[: int(fi.n_frames)]
        return out

    def _bottom_text_font_size(self, text: str, max_width: int) -> int:
        if not text:
            return max(5, int(round(5 * self._font_scale())))
        max_size = max(5, int(round(5 * self._font_scale())))
        min_size = max(1, int(round(1 * self._font_scale())))
        for size in range(max_size, min_size - 1, -1):
            if len(text) * size * 0.95 <= max_width:
                return size
        return min_size

    def _top_title_font_size(self, text: str, max_width: int) -> int:
        if not text:
            return max(4, int(round(6 * self._font_scale())))
        max_size = max(6, int(round(8 * self._font_scale())))
        min_size = max(2, int(round(2 * self._font_scale())))
        for size in range(max_size, min_size - 1, -1):
            if len(text) * size * 0.72 <= max_width:
                return size
        return min_size

    def _add_overlay(self) -> None:
        """Add text annotations and orientation axes to the plotter."""
        if self._plotter is None or self._vd is None:
            return
        source = self._vd.source_path
        if source:
            from pathlib import Path
            try:
                source_text = str(Path(source).resolve())
            except Exception:
                source_text = str(source)
        else:
            source_text = "-"
        fi = self._vd.field_info.get(self._active_field) if self._active_field else None
        field_text = self._field_label() or "-"
        frame_text = f"  {self._frame_label(fi)}" if fi else ""
        generated = self._vd.metadata.get("generated_field_info", {})
        diff_text = ""
        if self._active_field and isinstance(generated, dict):
            info = generated.get(self._active_field, {})
            if isinstance(info, dict):
                diff_text = str(info.get("overlay_text") or info.get("display_text") or "")
        title_text = f"{field_text}{frame_text}"
        if diff_text:
            title_text += f"\n{diff_text}"

        self._plotter.add_text(
            title_text,
            position="upper_left",
            font_size=int(round(11 * self._font_scale())),
            color="black",
            shadow=False,
        )
        self._plotter.add_text(
            source_text,
            position=(10, 10),
            font_size=self._bottom_text_font_size(source_text, max(1, self._window_size[0] - 20)),
            color="black",
            shadow=False,
        )
        # Orientation widget in the lower-left corner. The widget is a
        # VTK ``OrientationMarkerWidget`` driven by the active camera,
        # so it ALREADY rotates with the scene — we just need to flip
        # the labels back on (X / Y / Z) so the user can read which
        # axis is which while orbiting the model. The arrow colours
        # follow the conventional R/G/B = X/Y/Z so the labels stay
        # consistent with the strokes.
        try:
            self._plotter.add_axes(
                interactive=False,
                line_width=2,
                xlabel="X",
                ylabel="Y",
                zlabel="Z",
                x_color="#d4351c",
                y_color="#2da44e",
                z_color="#1f6feb",
                label_size=(0.4, 0.16),
                labels_off=False,
            )
        except TypeError:
            # Older PyVista builds may not accept every kwarg. Fall back
            # to the minimal labelled form so the user still sees X/Y/Z.
            try:
                self._plotter.add_axes(
                    interactive=False,
                    line_width=2,
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                )
            except TypeError:
                self._plotter.add_axes(interactive=False, line_width=2)

    # ------------------------------------------------------------------
    # Element selection / visibility
    # ------------------------------------------------------------------

    def select_element_at(self, x_px: int, y_px: int, *, operation: str = "add") -> dict:
        """Select or deselect one visible original element by screen pixel."""
        picked = self._pick_original_cell_at(x_px, y_px)
        hit = None
        if picked is not None:
            cell_id, marker = picked
            op = str(operation or "add").strip().lower()
            if op in ("remove", "subtract", "delete"):
                self._selected_cell_ids.discard(cell_id)
                selected = False
            elif op == "toggle" and cell_id in self._selected_cell_ids:
                self._selected_cell_ids.remove(cell_id)
                selected = False
            else:
                was_selected = cell_id in self._selected_cell_ids
                self._selected_cell_ids.add(cell_id)
                if not was_selected:
                    self._selection_undo_stack.append([int(cell_id)])
                selected = True
            hit = {"cell_id": int(cell_id), "selected": selected, "marker": marker}
        return self.element_selection_state(hit=hit)

    def select_elements_in_box(
        self,
        x0_px: int,
        y0_px: int,
        x1_px: int,
        y1_px: int,
        *,
        append: bool = True,
        operation: str = "add",
    ) -> dict:
        """Select visible original elements whose displayed cell centers fall in a box."""
        if self._plotter is None or self._grid is None:
            return self.element_selection_state(hit={"box_count": 0})
        display_grid = self._display_grid()
        if display_grid is None:
            return self.element_selection_state(hit={"box_count": 0})
        try:
            self._plotter.render()
            centers = display_grid.cell_centers().points
        except Exception:
            return self.element_selection_state(hit={"box_count": 0})
        xmin = min(int(x0_px), int(x1_px))
        xmax = max(int(x0_px), int(x1_px))
        ymin = min(int(y0_px), int(y1_px))
        ymax = max(int(y0_px), int(y1_px))
        selected: set[int] = set()
        for display_cell_id, center in enumerate(np.asarray(centers, dtype=np.float64)):
            try:
                projected = self._project_world_point((float(center[0]), float(center[1]), float(center[2])))
            except Exception:
                projected = None
            if projected is None:
                continue
            px, py = float(projected[0]), float(projected[1])
            if px < xmin or px > xmax or py < ymin or py > ymax:
                continue
            orig_cell_id = self._reverse_cell_id(display_grid, int(display_cell_id))
            if orig_cell_id is None:
                continue
            orig_cell_id = int(orig_cell_id)
            if orig_cell_id in self._hidden_cell_ids:
                continue
            selected.add(orig_cell_id)
        if not append:
            self._selected_cell_ids.clear()
        op = str(operation or "add").strip().lower()
        if op in ("remove", "subtract", "delete"):
            self._selected_cell_ids.difference_update(selected)
        else:
            newly_added = sorted(int(c) for c in selected if c not in self._selected_cell_ids)
            self._selected_cell_ids.update(selected)
            if newly_added:
                self._selection_undo_stack.append(newly_added)
        return self.element_selection_state(hit={"box_count": len(selected)})

    def select_elements_by_id(
        self,
        start_id: object = None,
        end_id: object = None,
        *,
        operation: str = "add",
    ) -> dict:
        """Select visible original elements by inclusive element-id range."""
        if self._grid is None:
            return self.element_selection_state(hit={"id_count": 0})

        def parse_int(value: object) -> Optional[int]:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            if text[0] in "+-":
                digits = text[1:]
            else:
                digits = text
            if not digits.isdigit():
                return None
            return int(text)

        start = parse_int(start_id)
        end = parse_int(end_id)
        if start is None and end is None:
            return self.element_selection_state(hit={"id_count": 0})
        if start is None:
            start = end
        if end is None:
            end = start
        if start is None or end is None:
            return self.element_selection_state(hit={"id_count": 0})

        n_cells = int(getattr(self._grid, "n_cells", 0) or 0)
        if n_cells <= 0:
            return self.element_selection_state(hit={"id_count": 0})
        raw_lo = min(int(start), int(end))
        raw_hi = max(int(start), int(end))
        lo = min(max(raw_lo, 0), n_cells - 1)
        hi = min(max(raw_hi, 0), n_cells - 1)
        if hi < lo:
            return self.element_selection_state(hit={"id_count": 0})
        selected = {
            int(cell_id)
            for cell_id in range(lo, hi + 1)
            if int(cell_id) not in self._hidden_cell_ids
        }
        op = str(operation or "add").strip().lower()
        if op in ("remove", "subtract", "delete"):
            self._selected_cell_ids.difference_update(selected)
        else:
            newly_added = sorted(int(c) for c in selected if c not in self._selected_cell_ids)
            self._selected_cell_ids.update(selected)
            if newly_added:
                self._selection_undo_stack.append(newly_added)
        return self.element_selection_state(hit={"id_count": len(selected)})

    def undo_element_selection(self) -> dict:
        """Undo the most recent selection append batch without affecting hidden cells."""
        while self._selection_undo_stack:
            restored = self._selection_undo_stack.pop()
            current = [int(c) for c in restored if int(c) in self._selected_cell_ids]
            if not current:
                continue
            for cell_id in current:
                self._selected_cell_ids.discard(cell_id)
            break
        return self.element_selection_state()

    def hide_selected_elements(self) -> dict:
        """Hide selected original elements from rendering and picking."""
        to_hide = sorted(int(c) for c in self._selected_cell_ids if c not in self._hidden_cell_ids)
        if to_hide:
            self._hidden_cell_ids.update(to_hide)
            self._hidden_undo_stack.append(to_hide)
            self._selected_cell_ids.clear()
            self._selection_undo_stack.clear()
            self._visible_grid_cache = None
            self._visible_grid_cache_key = None
            if not self._refresh_main_actor_in_place():
                self._rebuild_plotter()
        return self.element_selection_state()

    def show_only_selected_elements(self) -> dict:
        """Hide every currently visible element except the selected set."""
        if self._grid is None or not self._selected_cell_ids:
            return self.element_selection_state()
        n_cells = int(getattr(self._grid, "n_cells", 0) or 0)
        selected = {int(c) for c in self._selected_cell_ids}
        to_hide = [
            int(cell_id)
            for cell_id in range(n_cells)
            if int(cell_id) not in selected and int(cell_id) not in self._hidden_cell_ids
        ]
        if to_hide:
            self._hidden_cell_ids.update(to_hide)
            self._hidden_undo_stack.append(to_hide)
            self._visible_grid_cache = None
            self._visible_grid_cache_key = None
            if not self._refresh_main_actor_in_place():
                self._rebuild_plotter()
        return self.element_selection_state()

    def undo_hidden_elements(self) -> dict:
        """Restore the most recently hidden batch."""
        if self._hidden_undo_stack:
            restored = self._hidden_undo_stack.pop()
            for cell_id in restored:
                self._hidden_cell_ids.discard(int(cell_id))
            self._visible_grid_cache = None
            self._visible_grid_cache_key = None
            if not self._refresh_main_actor_in_place():
                self._rebuild_plotter()
        return self.element_selection_state()

    def restore_hidden_elements(self) -> dict:
        """Restore all hidden elements and clear the selection."""
        if self._hidden_cell_ids or self._selected_cell_ids or self._hidden_undo_stack:
            self._hidden_cell_ids.clear()
            self._selected_cell_ids.clear()
            self._selection_undo_stack.clear()
            self._hidden_undo_stack.clear()
            self._visible_grid_cache = None
            self._visible_grid_cache_key = None
            if not self._refresh_main_actor_in_place():
                self._rebuild_plotter()
        return self.element_selection_state()

    def clear_element_selection(self) -> dict:
        self._selected_cell_ids.clear()
        self._selection_undo_stack.clear()
        return self.element_selection_state()

    def element_selection_state(self, *, hit: Optional[dict] = None) -> dict:
        markers = []
        for cell_id in sorted(int(c) for c in self._selected_cell_ids):
            marker = self._element_marker(cell_id)
            if marker is not None:
                markers.append({"cell_id": cell_id, "marker": marker})
        return {
            "hit": hit,
            "selected": sorted(int(c) for c in self._selected_cell_ids),
            "selected_count": len(self._selected_cell_ids),
            "can_selection_undo": bool(self._selection_undo_stack),
            "hidden_count": len(self._hidden_cell_ids),
            "can_undo": bool(self._hidden_undo_stack),
            "total_count": int(getattr(self._grid, "n_cells", 0) or 0) if self._grid is not None else 0,
            "markers": markers,
        }

    def _pick_original_cell_at(self, x_px: int, y_px: int) -> Optional[tuple[int, Optional[dict]]]:
        """Return ``(original_cell_id, marker)`` for a visible cell hit."""
        if self._plotter is None or self._grid is None or self._main_actor is None:
            return None
        display_grid = self._display_grid()
        if display_grid is None:
            return None
        try:
            import vtk  # local import keeps PyVista-free environments importable
        except ImportError:
            return None
        try:
            self._plotter.render()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.InitializePickList()
            picker.AddPickList(self._main_actor)
            picker.PickFromListOn()
            win_h = int(self._window_size[1])
            vtk_y = max(0, win_h - int(y_px) - 1)
            picked = picker.Pick(int(x_px), int(vtk_y), 0, self._plotter.renderer)
            if not picked:
                return None
            display_cell_id = int(picker.GetCellId())
            if display_cell_id < 0:
                return None
            orig_cell_id = self._reverse_cell_id(display_grid, display_cell_id)
            if orig_cell_id is None:
                return None
            orig_cell_id = int(orig_cell_id)
            if orig_cell_id in self._hidden_cell_ids:
                return None
            return orig_cell_id, self._element_marker(orig_cell_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Probe (Abaqus-style picking)
    # ------------------------------------------------------------------

    def pick(
        self,
        x_px: int,
        y_px: int,
        mode: str = "node",
        *,
        include_frames: bool = False,
    ) -> Optional[dict]:
        """Pick a point or cell at viewport pixel ``(x_px, y_px)``.

        Two modes, matching the Abaqus "Probe Values" dialog:

        - ``mode='node'`` — only valid for node-located fields
          (``location == 'node'``). Returns the original-mesh node id,
          its world coordinates, the current scalar value at that
          node, plus the (informational) list of cells that share the
          node.
        - ``mode='element'`` — valid for gauss / element-located
          fields. Returns the single cell hit by the ray and the raw
          per-Gauss-point scalars on that cell, applying the same
          per-component / Mises / equivalent / magnitude rule the
          colour map uses.

        Returns ``None`` when:

        - no scene / no active field;
        - the active field's location does not match the requested
          probe mode (the frontend should already have constrained
          the field dropdown, but the server still guards);
        - the ray misses the mesh (background hit);
        - the scene is currently fully clipped away (no main actor).

        Implementation notes
        --------------------
        - Uses ``vtk.vtkCellPicker`` against the actually-rendered
          ``_display_grid()`` (possibly clipped). This is the surface
          the user sees, so picking is intuitive on cut faces.
        - VTK pixel space has its origin at the bottom-left of the
          render window, while the web canvas reports ``y`` growing
          downward from the top — we flip ``y_px`` accordingly.
        - The picker is restricted via ``PickFromListOn`` to
          ``self._main_actor`` so it cannot accidentally hit the
          scalar-bar / orientation widget / overlay text.
        - Cell ids are reverse-mapped via ``_orig_cell_id`` (cell_data
          attached in ``load()``) which survives ``clip()`` because
          cell_data is propagated to surviving cells unchanged. Node
          ids in node mode are NOT taken from the display grid's
          interpolated point id (interpolated points on cut faces
          have no real original counterpart); instead, we find the
          original cell that was hit, then the closest of its
          original nodes to the pick position.
        """
        if (
            self._plotter is None
            or self._vd is None
            or self._grid is None
            or self._main_actor is None
        ):
            return None
        if not self._active_field:
            return None
        fi = self._vd.field_info.get(self._active_field)
        if fi is None:
            return None

        normalized_mode = str(mode or "node").strip().lower()
        if normalized_mode == "node" and fi.location != "node":
            return None
        if normalized_mode == "element" and fi.location not in ("gauss", "element"):
            return None
        if normalized_mode not in ("node", "element"):
            return None

        display_grid = self._display_grid()
        if display_grid is None:
            return None

        try:
            import vtk  # local import keeps PyVista-free environments importable
        except ImportError:
            return None

        # Make sure the off-screen buffer is up to date before picking;
        # otherwise the z-buffer may still reflect the previous frame
        # and the ray hits stale geometry.
        try:
            self._plotter.render()
        except Exception:
            return None

        try:
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            picker.InitializePickList()
            picker.AddPickList(self._main_actor)
            picker.PickFromListOn()
        except Exception:
            return None

        # Web canvas Y grows downward; VTK render-window Y grows upward.
        win_h = int(self._window_size[1])
        vtk_y = max(0, win_h - int(y_px) - 1)
        try:
            picked = picker.Pick(int(x_px), int(vtk_y), 0, self._plotter.renderer)
        except Exception:
            return None
        if not picked:
            return None

        try:
            clipped_cell_id = int(picker.GetCellId())
        except Exception:
            return None
        if clipped_cell_id < 0:
            return None

        # Reverse-map cell id from display grid -> original grid.
        orig_cell_id = self._reverse_cell_id(display_grid, clipped_cell_id)
        if orig_cell_id is None:
            return None

        try:
            pick_xyz = tuple(float(v) for v in picker.GetPickPosition())
        except Exception:
            return None

        frame_time = self._current_frame_time()
        field_label = self._field_label()

        if normalized_mode == "node":
            nearest_pid, nearest_xyz = self._closest_original_node(orig_cell_id, pick_xyz)
            if nearest_pid is None:
                return None
            try:
                active = self._grid.point_data["active"]
                value = float(active[int(nearest_pid)])
            except Exception:
                value = float("nan")
            owning: List[int] = []
            if self._node_to_cells is not None:
                owning = [int(c) for c in self._node_to_cells.get(int(nearest_pid), [])]
            out = {
                "mode": "node",
                "point_id": int(nearest_pid),
                "world_xyz": [float(nearest_xyz[0]), float(nearest_xyz[1]), float(nearest_xyz[2])],
                "marker": self._node_marker(nearest_xyz),
                "value": value,
                "field_key": fi.key,
                "field_label": field_label,
                "component_index": self._active_component,
                "frame_idx": int(self._active_frame),
                "frame_time": frame_time,
                "owning_cell_ids": owning,
            }
            if include_frames:
                out["frame_series"] = self._node_frame_series(int(nearest_pid), fi)
            return out

        # element mode
        try:
            _, _, frame_data = self._resolve_field_data()
        except Exception:
            return None
        result = self._cell_gauss_values(orig_cell_id, fi, frame_data)
        if result is None:
            return None
        n_gauss, gauss_values = result
        centroid = self._cell_centroid(orig_cell_id)
        out = {
            "mode": "element",
            "cell_id": int(orig_cell_id),
            "centroid_xyz": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            "marker": self._element_marker(orig_cell_id),
            "n_gauss": int(n_gauss),
            "gauss_values": [float(v) for v in gauss_values],
            "field_key": fi.key,
            "field_label": field_label,
            "component_index": self._active_component,
            "frame_idx": int(self._active_frame),
            "frame_time": frame_time,
        }
        if include_frames:
            out["frame_series"] = self._element_frame_series(int(orig_cell_id), fi)
        return out

    def extrema_probes(
        self,
        n: int,
        *,
        include_frames: bool = True,
        exclude_nodes: Optional[list[int]] = None,
        exclude_cells: Optional[list[int]] = None,
        visible_only: bool = False,
    ) -> dict:
        """Return probe payloads for extrema of the active scalar field.

        Selection follows the GUI rule:
        - all finite values > 0: take the ``n`` largest values;
        - mixed positive/negative values: take round-half-up(n/2) largest
          positive values and the remaining most-negative values;
        - only non-positive values: take the ``n`` most-negative values;
        - if there are at most ``n`` finite values, return all finite values.

        Node fields generate node probe payloads. Gauss / element fields
        generate element probe payloads; for Gauss fields the selected
        extremum GP is stored as ``extreme_gauss_index`` while the card still
        carries all GP values for that element, matching manual element probe
        behavior.
        """
        if (
            self._vd is None
            or self._grid is None
            or not self._active_field
        ):
            return {"results": [], "reason": "no_scene"}
        fi = self._vd.field_info.get(self._active_field)
        if fi is None:
            return {"results": [], "reason": "no_field"}
        n = max(1, min(10, int(n or 1)))
        excluded_nodes = {int(v) for v in (exclude_nodes or []) if v is not None}
        excluded_cells = {int(v) for v in (exclude_cells or []) if v is not None}
        try:
            _, _, frame_data = self._resolve_field_data()
        except Exception:
            return {"results": [], "reason": "no_data"}

        candidates: list[dict] = []
        if fi.location == "node":
            try:
                scalars = np.asarray(self.get_scalar_data(), dtype=np.float64).reshape(-1)
                points = np.asarray(self._grid.points, dtype=np.float64)
            except Exception:
                return {"results": [], "reason": "no_data"}
            visible_nodes = self._visible_node_ids() if visible_only else None
            limit = min(int(scalars.size), int(points.shape[0]))
            for point_id in range(limit):
                if int(point_id) in excluded_nodes:
                    continue
                if visible_nodes is not None and point_id not in visible_nodes:
                    continue
                value = float(scalars[point_id])
                if not np.isfinite(value):
                    continue
                candidates.append({
                    "kind": "node",
                    "point_id": int(point_id),
                    "value": value,
                    "world_xyz": (
                        float(points[point_id][0]),
                        float(points[point_id][1]),
                        float(points[point_id][2]),
                    ),
                })
        elif fi.location in {"gauss", "element"}:
            n_cells = int(getattr(self._grid, "n_cells", 0) or 0)
            for cell_id in range(n_cells):
                if int(cell_id) in excluded_cells:
                    continue
                if visible_only and int(cell_id) in self._hidden_cell_ids:
                    continue
                values = self._cell_gauss_values(int(cell_id), fi, frame_data)
                if values is None:
                    continue
                n_gauss, gauss_values = values
                for gp_index, value in enumerate(gauss_values):
                    value = float(value)
                    if not np.isfinite(value):
                        continue
                    candidates.append({
                        "kind": "element",
                        "cell_id": int(cell_id),
                        "gp_index": int(gp_index),
                        "n_gauss": int(n_gauss),
                        "value": value,
                    })
        else:
            return {"results": [], "reason": "unsupported_location"}

        chosen = self._choose_extrema_candidates(candidates, n)
        results = [
            self._probe_from_extreme_candidate(
                item,
                fi,
                include_frames=include_frames,
                allow_hidden=not visible_only,
            )
            for item in chosen
        ]
        return {
            "results": [r for r in results if r is not None],
            "requested": int(n),
            "available": int(len(candidates)),
            "field_key": fi.key,
            "field_label": self._field_label(),
            "location": fi.location,
            "visible_only": bool(visible_only),
        }

    # ---- probe helpers -----------------------------------------------

    def _choose_extrema_candidates(self, candidates: list[dict], n: int) -> list[dict]:
        if not candidates:
            return []
        n = max(1, int(n))

        def key_of(item: dict) -> tuple[str, int]:
            if item.get("kind") == "node":
                return ("node", int(item.get("point_id", -1)))
            return ("element", int(item.get("cell_id", -1)))

        def take_unique(items: list[dict], count: int, used: set[tuple[str, int]]) -> list[dict]:
            out: list[dict] = []
            for item in items:
                key = key_of(item)
                if key in used:
                    continue
                used.add(key)
                out.append(item)
                if len(out) >= count:
                    break
            return out

        if len(candidates) <= n:
            return take_unique(
                sorted(candidates, key=lambda c: float(c.get("value", 0.0)), reverse=True),
                n,
                set(),
            )
        positives = [c for c in candidates if float(c.get("value", 0.0)) > 0.0]
        negatives = [c for c in candidates if float(c.get("value", 0.0)) < 0.0]
        if len(positives) == len(candidates):
            return take_unique(
                sorted(candidates, key=lambda c: float(c["value"]), reverse=True),
                n,
                set(),
            )
        if positives and negatives:
            n_pos = int(np.floor(float(n) / 2.0 + 0.5))
            n_neg = n - n_pos
            used: set[tuple[str, int]] = set()
            pos = take_unique(
                sorted(positives, key=lambda c: float(c["value"]), reverse=True),
                n_pos,
                used,
            )
            neg = take_unique(
                sorted(negatives, key=lambda c: float(c["value"])),
                n_neg,
                used,
            )
            return pos + neg
        if negatives:
            return take_unique(
                sorted(negatives, key=lambda c: float(c["value"])),
                n,
                set(),
            )
        return take_unique(
            sorted(candidates, key=lambda c: abs(float(c.get("value", 0.0))), reverse=True),
            n,
            set(),
        )

    def _visible_node_ids(self) -> Optional[set[int]]:
        if not self._hidden_cell_ids:
            return None
        visible: set[int] = set()
        if self._grid is None:
            return visible
        try:
            n_cells = int(self._grid.n_cells)
            for cell_id in range(n_cells):
                if int(cell_id) in self._hidden_cell_ids:
                    continue
                cell = self._grid.GetCell(int(cell_id))
                for i in range(int(cell.GetNumberOfPoints())):
                    visible.add(int(cell.GetPointId(i)))
        except Exception:
            return None
        return visible

    def _probe_from_extreme_candidate(
        self,
        item: dict,
        fi: FieldInfo,
        *,
        include_frames: bool,
        allow_hidden: bool = False,
    ) -> Optional[dict]:
        frame_time = self._current_frame_time()
        field_label = self._field_label()
        if item.get("kind") == "node":
            point_id = int(item["point_id"])
            xyz = item.get("world_xyz", (0.0, 0.0, 0.0))
            owning: List[int] = []
            if self._node_to_cells is not None:
                owning = [int(c) for c in self._node_to_cells.get(point_id, [])]
            out = {
                "mode": "node",
                "point_id": point_id,
                "world_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "marker": self._node_marker((float(xyz[0]), float(xyz[1]), float(xyz[2]))),
                "value": float(item["value"]),
                "field_key": fi.key,
                "field_label": field_label,
                "component_index": self._active_component,
                "frame_idx": int(self._active_frame),
                "frame_time": frame_time,
                "owning_cell_ids": owning,
                "extreme_value": float(item["value"]),
            }
            if include_frames:
                out["frame_series"] = self._node_frame_series(point_id, fi)
            return out
        if item.get("kind") == "element":
            cell_id = int(item["cell_id"])
            if cell_id in self._hidden_cell_ids and not allow_hidden:
                return None
            try:
                _, _, frame_data = self._resolve_field_data()
            except Exception:
                return None
            values = self._cell_gauss_values(cell_id, fi, frame_data)
            if values is None:
                return None
            n_gauss, gauss_values = values
            centroid = self._cell_centroid(cell_id)
            gp_index = int(item.get("gp_index", 0))
            out = {
                "mode": "element",
                "cell_id": cell_id,
                "centroid_xyz": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "marker": self._element_marker(cell_id),
                "n_gauss": int(n_gauss),
                "gauss_values": [float(v) for v in gauss_values],
                "field_key": fi.key,
                "field_label": field_label,
                "component_index": self._active_component,
                "frame_idx": int(self._active_frame),
                "frame_time": frame_time,
                "extreme_value": float(item["value"]),
                "extreme_gauss_index": gp_index + 1,
            }
            if include_frames:
                out["frame_series"] = self._element_frame_series(cell_id, fi)
            return out
        return None

    def _project_world_point(self, xyz: Tuple[float, float, float]) -> Optional[list[float]]:
        """Project a world point into web render-window pixels."""
        if self._plotter is None:
            return None
        try:
            renderer = self._plotter.renderer
            renderer.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            win_h = int(self._window_size[1])
            return [float(x), float(max(0, win_h - float(y) - 1.0))]
        except Exception:
            return None

    def _node_marker(self, xyz: Tuple[float, float, float]) -> Optional[dict]:
        xy = self._project_world_point(xyz)
        if xy is None:
            return None
        return {"type": "node", "render_xy": xy, "label_xy": xy}

    def _element_marker(self, cell_id: int) -> Optional[dict]:
        if self._grid is None:
            return None
        if int(cell_id) in self._hidden_cell_ids:
            return None
        try:
            cell = self._grid.GetCell(int(cell_id))
            edges: list[list[list[float]]] = []
            pts: list[Tuple[float, float, float]] = []
            for edge_idx in range(int(cell.GetNumberOfEdges())):
                edge = cell.GetEdge(edge_idx)
                if edge is None or edge.GetNumberOfPoints() < 2:
                    continue
                p0 = self._grid.GetPoint(int(edge.GetPointId(0)))
                p1 = self._grid.GetPoint(int(edge.GetPointId(1)))
                pts.append((float(p0[0]), float(p0[1]), float(p0[2])))
                pts.append((float(p1[0]), float(p1[1]), float(p1[2])))
                q0 = self._project_world_point((float(p0[0]), float(p0[1]), float(p0[2])))
                q1 = self._project_world_point((float(p1[0]), float(p1[1]), float(p1[2])))
                if q0 is not None and q1 is not None:
                    edges.append([q0, q1])
            if edges:
                label_xy = None
                if pts:
                    arr = np.asarray(pts, dtype=np.float64)
                    c = arr.mean(axis=0)
                    label_xy = self._project_world_point((float(c[0]), float(c[1]), float(c[2])))
                return {"type": "element", "render_edges": edges, "label_xy": label_xy}
        except Exception:
            return None
        return None

    def project_probe_marker(self, marker: dict) -> Optional[dict]:
        """Re-project a saved probe target using the current camera.

        The frontend stores pinned probe cards as original node/cell ids
        plus world coordinates. After orbit/pan/zoom/view changes it asks
        the scene to re-project those saved targets so the red marker and
        probe number stay attached to the geometry instead of remaining
        fixed at the old screen pixel.
        """
        if self._grid is None:
            return None
        mode = str((marker or {}).get("mode", "")).lower()
        if mode == "node":
            xyz = marker.get("world_xyz")
            if not (isinstance(xyz, (list, tuple)) and len(xyz) >= 3):
                try:
                    pid = int(marker.get("point_id"))
                    p = self._grid.GetPoint(pid)
                    xyz = (float(p[0]), float(p[1]), float(p[2]))
                except Exception:
                    return None
            return self._node_marker((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        if mode == "element":
            try:
                return self._element_marker(int(marker.get("cell_id")))
            except Exception:
                return None
        return None

    def _frame_count_for_field(self, fi: FieldInfo) -> int:
        try:
            return max(1, int(fi.n_frames))
        except Exception:
            return 1

    def _frame_time_at(self, frame_idx: int) -> Optional[float]:
        if self._vd is None or self._vd.frame_times is None:
            return None
        ft = np.asarray(self._vd.frame_times, dtype=np.float64).reshape(-1)
        if ft.size == 0:
            return None
        idx = max(0, min(int(frame_idx), int(ft.size) - 1))
        try:
            return float(ft[idx])
        except Exception:
            return None

    def _node_frame_series(self, point_id: int, fi: FieldInfo) -> list[dict]:
        """Return scalar values for one picked node across all frames."""
        out: list[dict] = []
        n_frames = self._frame_count_for_field(fi)
        for frame_idx in range(n_frames):
            try:
                scalars = self.get_scalar_data(frame=frame_idx)
                value = float(np.asarray(scalars, dtype=np.float64).reshape(-1)[int(point_id)])
            except Exception:
                value = float("nan")
            out.append({
                "frame_idx": int(frame_idx),
                "frame_time": self._frame_time_at(frame_idx),
                "value": value,
            })
        return out

    def _element_frame_series(self, cell_id: int, fi: FieldInfo) -> list[dict]:
        """Return per-GP scalar values for one picked element across all frames."""
        out: list[dict] = []
        n_frames = self._frame_count_for_field(fi)
        for frame_idx in range(n_frames):
            try:
                _, _, frame_data = self._resolve_field_data(frame=frame_idx)
                values = self._cell_gauss_values(cell_id, fi, frame_data)
            except Exception:
                values = None
            if values is None:
                n_gauss, gauss_values = 0, []
            else:
                n_gauss, gauss_values = values
            out.append({
                "frame_idx": int(frame_idx),
                "frame_time": self._frame_time_at(frame_idx),
                "n_gauss": int(n_gauss),
                "gauss_values": [float(v) for v in gauss_values],
            })
        return out

    def _build_node_to_cells(self) -> Dict[int, List[int]]:
        """Inverse index: original node_id -> list of original cell_ids.

        Walks the VTK-packed cells array on ``self._grid`` exactly once;
        per node we keep a python list so the typical "owning elements"
        lookup is O(1) at probe time.
        """
        out: Dict[int, List[int]] = {}
        if self._grid is None:
            return out
        try:
            cells = np.asarray(self._grid.cells)
            n_cells = int(self._grid.n_cells)
        except Exception:
            return out
        pos = 0
        for cid in range(n_cells):
            if pos >= cells.size:
                break
            npe = int(cells[pos])
            nids = cells[pos + 1: pos + 1 + npe]
            for n in nids:
                out.setdefault(int(n), []).append(int(cid))
            pos += 1 + npe
        return out

    def _reverse_cell_id(self, display_grid, clipped_cell_id: int) -> Optional[int]:
        """Map a display-grid cell id back to the original-grid cell id.

        Prefers our own ``_orig_cell_id`` tag (attached in ``load()``).
        Falls back to PyVista's ``vtkOriginalCellIds`` if present, and
        finally to the identity mapping (no clip).
        """
        try:
            arr = display_grid.cell_data.get("_orig_cell_id")
        except Exception:
            arr = None
        if arr is None:
            try:
                arr = display_grid.cell_data.get("vtkOriginalCellIds")
            except Exception:
                arr = None
        if arr is None:
            orig = clipped_cell_id
        else:
            arr_np = np.asarray(arr)
            if clipped_cell_id < 0 or clipped_cell_id >= arr_np.size:
                return None
            orig = int(arr_np[clipped_cell_id])
        if orig < 0 or orig >= int(self._grid.n_cells):
            return None
        return orig

    def _closest_original_node(
        self, cell_id: int, world_xyz: tuple
    ) -> Tuple[Optional[int], Tuple[float, float, float]]:
        """Closest node of an original cell to the pick world position."""
        if self._grid is None:
            return None, (0.0, 0.0, 0.0)
        try:
            cell = self._grid.GetCell(int(cell_id))
            n = cell.GetNumberOfPoints()
            ids = [int(cell.GetPointId(i)) for i in range(n)]
        except Exception:
            return None, (0.0, 0.0, 0.0)
        if not ids:
            return None, (0.0, 0.0, 0.0)
        pts = np.asarray(self._grid.points)[ids]
        target = np.asarray(world_xyz, dtype=np.float64)
        d2 = np.sum((pts - target) ** 2, axis=1)
        k = int(np.argmin(d2))
        nid = ids[k]
        coords = (float(pts[k][0]), float(pts[k][1]), float(pts[k][2]))
        return nid, coords

    def _cell_centroid(self, cell_id: int) -> Tuple[float, float, float]:
        """Centroid of an original-grid cell (mean of its node coords)."""
        if self._grid is None:
            return (0.0, 0.0, 0.0)
        try:
            cell = self._grid.GetCell(int(cell_id))
            n = cell.GetNumberOfPoints()
            ids = [int(cell.GetPointId(i)) for i in range(n)]
            pts = np.asarray(self._grid.points)[ids]
            c = pts.mean(axis=0)
            return (float(c[0]), float(c[1]), float(c[2]))
        except Exception:
            return (0.0, 0.0, 0.0)

    def _cell_gauss_values(
        self, cell_id: int, fi: FieldInfo, frame_data: np.ndarray
    ) -> Optional[Tuple[int, List[float]]]:
        """Return ``(n_gauss, [scalar_per_gp])`` for one element.

        For ``location == 'gauss'`` we slice the flat
        ``[n_gauss_total, n_comp]`` block for the picked cell and run
        the same per-component / Mises / equivalent reduction that
        ``scalarize_field`` uses for the screen colour map, just
        without the gauss-to-node averaging step.

        For ``location == 'element'`` there is one value per cell, so
        ``n_gauss = 1``.
        """
        from jaxmech.modules.visualization.mesh_builder import (
            _elem_offsets,
            _normalize_gauss_field_layout,
        )

        try:
            if fi.location == "gauss":
                ngpe = _elem_offsets(self._vd)
                if cell_id < 0 or cell_id >= ngpe.size:
                    return None
                normalized = _normalize_gauss_field_layout(
                    np.asarray(frame_data, dtype=np.float64), self._vd
                )
                if normalized.ndim == 1:
                    normalized = normalized[:, np.newaxis]
                start = int(np.sum(ngpe[:cell_id]))
                n_gp = int(ngpe[cell_id])
                if n_gp <= 0 or start + n_gp > normalized.shape[0]:
                    return None
                block = normalized[start:start + n_gp]  # (n_gp, n_comp)
                scalars = scalarize_per_gp(
                    block, self._vd, location="gauss",
                    component=self._active_component,
                    field_key=fi.key,
                )
                vals = np.asarray(scalars, dtype=np.float64).reshape(-1)
                return n_gp, [float(v) for v in vals]
            if fi.location == "element":
                arr = np.asarray(frame_data, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                if cell_id < 0 or cell_id >= arr.shape[0]:
                    return None
                block = arr[cell_id:cell_id + 1]  # (1, n_comp)
                scalars = scalarize_per_gp(
                    block, self._vd, location="element",
                    component=self._active_component,
                    field_key=fi.key,
                )
                v = float(np.asarray(scalars, dtype=np.float64).reshape(-1)[0])
                return 1, [v]
        except Exception:
            return None
        return None

    def _current_frame_time(self) -> Optional[float]:
        if self._vd is None or self._vd.frame_times is None:
            return None
        ft = np.asarray(self._vd.frame_times, dtype=np.float64).reshape(-1)
        if ft.size == 0:
            return None
        idx = max(0, min(int(self._active_frame), int(ft.size) - 1))
        try:
            return float(ft[idx])
        except Exception:
            return None

    def _mat_identity(self) -> dict:
        """Stable identifier dict for the loaded MAT.

        Combines ``source_path`` + file mtime (ISO 8601, UTC) + size in
        bytes + manifest ``producer_module`` / ``analysis_type`` (when
        present). Used by the probe-card CSV export so a downstream
        consumer can unambiguously identify which physical MAT
        produced each row, even if multiple runs of the same model
        share a filename.
        """
        out: dict = {}
        if self._vd is None:
            return out
        src = str(self._vd.source_path or "")
        if src:
            out["source_path"] = src
            try:
                p = Path(src)
                if p.exists():
                    st = p.stat()
                    out["mtime_iso"] = datetime.fromtimestamp(
                        st.st_mtime, tz=timezone.utc
                    ).isoformat()
                    out["size_bytes"] = int(st.st_size)
            except Exception:
                pass
        meta = self._vd.metadata or {}
        if isinstance(meta, dict):
            for key in ("producer_module", "analysis_type"):
                value = meta.get(key, "")
                if value:
                    out[key] = str(value)
        return out

    def close(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None
