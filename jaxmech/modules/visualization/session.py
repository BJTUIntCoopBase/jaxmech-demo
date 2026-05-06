"""Visualisation session manager.

A session owns one or two ``VizScene`` instances (for single-view and
side-by-side comparison).  The web router delegates all state mutation
to the session so that camera / field state is kept between requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from jaxmech.modules.visualization.mat_reader import (
    VizData,
    load_viz_data,
    validation_counterpart_fields,
)
from jaxmech.modules.visualization.scene import VizScene


class VizSession:
    """Stateful visualisation session with up to two scenes."""

    def __init__(self, window_size: tuple[int, int] = (800, 600)):
        self._window_size = window_size
        self._scene_a: Optional[VizScene] = None
        self._scene_b: Optional[VizScene] = None
        self._vd_a: Optional[VizData] = None
        self._vd_b: Optional[VizData] = None
        self._compare_mode: bool = False
        self._compare_locked: bool = False
        self._sync_camera: bool = False

    # ------------------------------------------------------------------
    # Scene A (primary)
    # ------------------------------------------------------------------

    def load_a(self, mat_path: str | Path, selected_fields: Optional[Sequence[str]] = None) -> dict:
        """Load a .mat into scene A and return scene info."""
        self._compare_locked = False
        vd = load_viz_data(mat_path, selected_fields=selected_fields)
        self._vd_a = vd
        if self._scene_a is None:
            self._scene_a = VizScene(self._window_size)
        self._scene_a.load(vd)
        return self._scene_a.info

    @property
    def scene_a(self) -> Optional[VizScene]:
        return self._scene_a

    # ------------------------------------------------------------------
    # Scene B (comparison)
    # ------------------------------------------------------------------

    def load_b(self, mat_path: str | Path, selected_fields: Optional[Sequence[str]] = None) -> dict:
        """Load a .mat into scene B (comparison) and return scene info."""
        self._compare_locked = False
        vd = load_viz_data(mat_path, selected_fields=selected_fields)
        self._vd_b = vd
        if self._scene_b is None:
            self._scene_b = VizScene(self._window_size)
        self._scene_b.load(vd)
        self._compare_mode = True
        return self._scene_b.info

    def load_validation_pair(self, mat_path: str | Path, selected_fields: Optional[Sequence[str]] = None) -> dict:
        """Load an embedded ODB validation MAT as JAX-vs-ABAQUS A/B scenes."""
        fields_a = list(selected_fields or [])
        fields_b = validation_counterpart_fields(fields_a, target_side="abaqus")
        vd_a = load_viz_data(mat_path, selected_fields=fields_a, validation_side="jax")
        vd_b = load_viz_data(mat_path, selected_fields=fields_b, validation_side="abaqus")
        self._vd_a = vd_a
        self._vd_b = vd_b
        if self._scene_a is None:
            self._scene_a = VizScene(self._window_size)
        if self._scene_b is None:
            self._scene_b = VizScene(self._window_size)
        self._scene_a.load(vd_a)
        self._scene_b.load(vd_b)
        self._compare_mode = True
        self._compare_locked = True
        return {
            "validation_compare": True,
            "validation_compare_mode": "embedded_odb",
            "compare_locked": True,
            "scene_a": self._scene_a.info,
            "scene_b": self._scene_b.info,
        }

    @property
    def scene_b(self) -> Optional[VizScene]:
        return self._scene_b

    # ------------------------------------------------------------------
    # Compare mode
    # ------------------------------------------------------------------

    @property
    def compare_mode(self) -> bool:
        return self._compare_mode

    @property
    def compare_locked(self) -> bool:
        return self._compare_locked

    def set_compare_mode(self, enabled: bool) -> None:
        if self._compare_locked and not enabled:
            self._compare_mode = True
            return
        self._compare_mode = enabled

    def close_b(self, *, force: bool = False) -> None:
        """Close scene B and exit compare mode."""
        if self._compare_locked and not force:
            self._compare_mode = True
            return
        if self._scene_b is not None:
            self._scene_b.close()
            self._scene_b = None
        self._vd_b = None
        self._compare_mode = False
        self._compare_locked = False

    # ------------------------------------------------------------------
    # Synchronized camera
    # ------------------------------------------------------------------

    @property
    def sync_camera(self) -> bool:
        return self._sync_camera

    def set_sync_camera(self, enabled: bool) -> None:
        self._sync_camera = enabled
        if enabled:
            self.mirror_camera("a")

    def mirror_camera(self, source_slot: str = "a") -> None:
        """Copy camera from *source_slot* to the other scene."""
        src = self.get_scene(source_slot)
        dst = self.get_scene("b" if source_slot == "a" else "a")
        if src is None or dst is None:
            return
        cam = src.get_camera_state()
        if cam:
            dst.set_camera_state(cam)

    # ------------------------------------------------------------------
    # Convenience: get scene by slot name
    # ------------------------------------------------------------------

    def get_scene(self, slot: str = "a") -> Optional[VizScene]:
        return self._scene_a if slot == "a" else self._scene_b

    def get_vd(self, slot: str = "a") -> Optional[VizData]:
        return self._vd_a if slot == "a" else self._vd_b

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict:
        return {
            "compare_mode": self._compare_mode,
            "compare_locked": self._compare_locked,
            "sync_camera": self._sync_camera,
            "scene_a": self._scene_a.info if self._scene_a else None,
            "scene_b": self._scene_b.info if self._scene_b else None,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        for s in (self._scene_a, self._scene_b):
            if s is not None:
                s.close()
        self._scene_a = None
        self._scene_b = None
        self._vd_a = None
        self._vd_b = None
        self._compare_locked = False
