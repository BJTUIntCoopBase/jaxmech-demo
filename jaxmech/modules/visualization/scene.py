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
from typing import Optional, Tuple

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
def _component_tag(fi: FieldInfo, comp: int) -> str:
    """Return a human-readable component tag for the colorbar title."""
    key_lower = fi.key.lower()
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    nc = fi.n_components
    is_stress = "stress" in text
    is_strain = "strain" in text
    is_disp = "displacement" in text or key_lower in {
        "u",
        "frame_u",
        "elastic_u",
        "solid_u_nodal",
    }
    is_force = "nforc" in text or "reaction" in text or "force" in text or key_lower in {
        "shakedown_equality_violation",
    }
    if is_stress:
        labels = _STRESS_COMP_6 if nc == 6 else (_STRESS_COMP_3 if nc == 3 else None)
        if labels and 0 <= comp < len(labels):
            return labels[comp]
    elif is_strain:
        labels = _STRAIN_COMP_6 if nc == 6 else (_STRAIN_COMP_3 if nc == 3 else None)
        if labels and 0 <= comp < len(labels):
            return labels[comp]
    elif is_disp:
        labels = _DISP_COMP
        if 0 <= comp < len(labels):
            return labels[comp]
    elif is_force:
        labels = _FORCE_COMP_6 if nc == 6 else _FORCE_COMP
        if 0 <= comp < len(labels):
            return labels[comp]
    return f"comp {comp}"


def _default_scalar_tag(fi: Optional[FieldInfo]) -> str:
    if fi is None or fi.n_components <= 1:
        return ""
    key_lower = fi.key.lower()
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    if "stress" in text:
        return "Mises"
    if "strain" in text:
        return "Equivalent"
    if (
        key_lower in {"u", "frame_u", "elastic_u", "solid_u_nodal"}
        or "displacement" in text
        or "nforc" in text
        or "reaction" in text
        or "force" in text
        or key_lower in {"shakedown_equality_violation"}
    ):
        return "Magnitude"
    return ""


def _field_base_label(fi: Optional[FieldInfo]) -> str:
    if fi is None:
        return ""
    key_lower = fi.key.lower()
    label_lower = (fi.label or "").lower()
    text = f"{key_lower} {label_lower}"
    if "stress" in text:
        return "Stress"
    if "strain" in text:
        return "Strain"
    if key_lower in {"u", "frame_u", "elastic_u", "solid_u_nodal"} or "displacement" in text:
        return "Displacement"
    if "nforc" in text or "reaction" in text or "force" in text or key_lower == "shakedown_equality_violation":
        return "NFORC"
    return fi.label or fi.key


DEFAULT_WINDOW_SIZE = (800, 600)
STREAM_QUALITY = 85  # JPEG quality for WebSocket frames
_UNSET = object()


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
        self._show_edges: bool = False
        self._deform_scale: float = 0.0  # 0 = off
        self._show_overlay: bool = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, vd: VizData) -> list[str]:
        """Load visualisation data and return available field keys."""
        self._vd = vd
        self._grid = build_grid(vd)
        self._base_points = np.array(self._grid.points, copy=True)
        self._active_field = None
        self._active_frame = 0
        self._clim = None
        self._deform_scale = 0.0
        self._rebuild_plotter()

        keys = vd.field_keys()
        if keys:
            self.set_field(keys[0])
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
            "show_edges": self._show_edges,
            "show_overlay": self._show_overlay,
            "deform_scale": self._deform_scale,
            "has_displacement": has_disp,
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

        if self._grid is not None and "active" in self._grid.point_data:
            font_scale = self._font_scale()
            kwargs: dict = {
                "scalars": "active",
                "cmap": self._cmap,
                "show_edges": self._show_edges,
                "scalar_bar_args": {
                    "title": "",
                    "color": "black",
                    "title_font_size": int(round(11 * font_scale)),
                    "label_font_size": int(round(10 * font_scale)),
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
            self._plotter.add_mesh(self._grid, **kwargs)
            self._add_scalar_bar_title()
        elif self._grid is not None:
            self._plotter.add_mesh(
                self._grid,
                color="lightgray",
                show_edges=True,
            )

        if self._show_overlay:
            self._add_overlay()

        if cam_state is not None:
            self.set_camera_state(cam_state)
        else:
            self.set_preset_view("iso")

    def _font_scale(self) -> float:
        width, height = self._window_size
        return max(1.0, min(width / DEFAULT_WINDOW_SIZE[0], height / DEFAULT_WINDOW_SIZE[1]))

    def _fit_camera_to_grid(self, margin: float = 1.08, zoom: float = 1.35) -> None:
        """Fit the current camera orientation to the grid with a small margin."""
        if self._plotter is None or self._grid is None:
            return
        try:
            self._plotter.reset_camera(bounds=self._grid.bounds)
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

    def _add_scalar_bar_title(self) -> None:
        """Place the scalar title manually so it stays clear of numeric labels."""
        if self._plotter is None:
            return
        title = self._field_label()
        if not title:
            return
        width, height = self._window_size
        x = max(8, int(width * 0.08))
        y = min(height - 35, int(height * (0.22 + 0.58)) + 34)
        self._plotter.add_text(
            title,
            position=(x, y),
            font_size=int(round(9 * self._font_scale())),
            color="black",
            shadow=False,
        )

    def _field_label(self) -> str:
        if self._active_field is None or self._vd is None:
            return ""
        fi = self._vd.field_info.get(self._active_field)
        if fi is None:
            return self._active_field
        label = _field_base_label(fi)
        if self._active_component is not None:
            comp = self._active_component
            tag = _component_tag(fi, comp)
            label += f" - {tag}"
        else:
            default_tag = _default_scalar_tag(fi)
            if default_tag:
                label += f" - {default_tag}"
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
            font_size=int(round(9 * self._font_scale())),
            color="black",
            shadow=True,
        )
        self._plotter.add_text(
            source_text,
            position=(10, 10),
            font_size=self._bottom_text_font_size(source_text, max(1, self._window_size[0] - 20)),
            color="black",
            shadow=True,
        )
        try:
            self._plotter.add_axes(
                interactive=False,
                line_width=2,
                color="black",
                labels_off=True,
            )
        except TypeError:
            self._plotter.add_axes(
                interactive=False,
                line_width=2,
                color="black",
            )

    def close(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None
