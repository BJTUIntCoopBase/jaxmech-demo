"""Visualization session manager for one or two demo MAT scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from jaxmech.modules.visualization.mat_reader import VizData, load_viz_data
from jaxmech.modules.visualization.scene import VizScene


class VizSession:
    """Stateful visualization session with optional A/B comparison."""

    def __init__(self, window_size: tuple[int, int] = (800, 600)):
        self._window_size = window_size
        self._scene_a: Optional[VizScene] = None
        self._scene_b: Optional[VizScene] = None
        self._vd_a: Optional[VizData] = None
        self._vd_b: Optional[VizData] = None
        self._compare_mode = False
        self._sync_camera = False

    def load_a(self, mat_path: str | Path, selected_fields: Optional[Sequence[str]] = None) -> dict:
        vd = load_viz_data(mat_path, selected_fields=selected_fields)
        self._vd_a = vd
        if self._scene_a is None:
            self._scene_a = VizScene(self._window_size)
        self._scene_a.load(vd)
        return self._scene_a.info

    def load_b(self, mat_path: str | Path, selected_fields: Optional[Sequence[str]] = None) -> dict:
        vd = load_viz_data(mat_path, selected_fields=selected_fields)
        self._vd_b = vd
        if self._scene_b is None:
            self._scene_b = VizScene(self._window_size)
        self._scene_b.load(vd)
        self._compare_mode = True
        return self._scene_b.info

    def close_b(self) -> None:
        if self._scene_b is not None:
            self._scene_b.close()
        self._scene_b = None
        self._vd_b = None
        self._compare_mode = False

    @property
    def compare_mode(self) -> bool:
        return self._compare_mode

    def set_compare_mode(self, enabled: bool) -> None:
        self._compare_mode = bool(enabled)

    @property
    def sync_camera(self) -> bool:
        return self._sync_camera

    def set_sync_camera(self, enabled: bool) -> None:
        self._sync_camera = bool(enabled)
        if enabled:
            self.mirror_camera("a")

    def mirror_camera(self, source_slot: str = "a") -> None:
        src = self.get_scene(source_slot)
        dst = self.get_scene("b" if source_slot == "a" else "a")
        if src is not None and dst is not None:
            cam = src.get_camera_state()
            if cam:
                dst.set_camera_state(cam)

    def get_scene(self, slot: str = "a") -> Optional[VizScene]:
        return self._scene_a if slot == "a" else self._scene_b

    def get_vd(self, slot: str = "a") -> Optional[VizData]:
        return self._vd_a if slot == "a" else self._vd_b

    @property
    def info(self) -> dict:
        return {
            "compare_mode": self._compare_mode,
            "compare_locked": False,
            "sync_camera": self._sync_camera,
            "scene_a": self._scene_a.info if self._scene_a else None,
            "scene_b": self._scene_b.info if self._scene_b else None,
        }

    def close(self) -> None:
        for scene in (self._scene_a, self._scene_b):
            if scene is not None:
                scene.close()
        self._scene_a = None
        self._scene_b = None
        self._vd_a = None
        self._vd_b = None
