"""High-resolution screenshot export (300 DPI PNG).

PyVista's ``screenshot(scale=N)`` multiplies pixel dimensions.  To
produce a 300-DPI publication-quality image we:

1. Render at a chosen *scale* factor (default 4x).
2. Use Pillow to embed the DPI metadata in the PNG.
3. Optionally crop / pad to an exact physical size.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None  # type: ignore[assignment]

from jaxmech.modules.visualization.scene import VizScene

DEFAULT_DPI = 300
DEFAULT_SCALE = 4


def screenshot_bytes(
    scene: VizScene,
    *,
    scale: int = DEFAULT_SCALE,
    dpi: int = DEFAULT_DPI,
    transparent_background: bool = False,
) -> bytes:
    """Render the scene at high resolution and return PNG bytes with DPI metadata.

    Parameters
    ----------
    scene : VizScene
        The scene to capture.
    scale : int
        Pixel multiplier applied to the plotter window size.
    dpi : int
        DPI value to embed in the PNG metadata.
    transparent_background : bool
        If True, render with a transparent background.

    Returns
    -------
    bytes
        PNG image data.
    """
    from PIL import Image

    if scene._plotter is None:
        raise RuntimeError("Scene has no active plotter.")

    old_size = scene._window_size
    old_camera = scene.get_camera_state()
    old_bg = None
    try:
        scene._window_size = (
            max(1, int(old_size[0] * max(1, scale))),
            max(1, int(old_size[1] * max(1, scale))),
        )
        scene._rebuild_plotter()
        if old_camera:
            scene.set_camera_state(old_camera)
        plotter = scene._plotter
        if plotter is None:
            raise RuntimeError("Scene has no active plotter.")

        if transparent_background:
            old_bg = plotter.background_color
            plotter.set_background("white", top="white")

        img_arr = plotter.screenshot(
            return_img=True,
            scale=1,
            transparent_background=transparent_background,
        )

        if old_bg is not None:
            plotter.set_background(old_bg)
    finally:
        scene._window_size = old_size
        scene._rebuild_plotter()
        if old_camera:
            scene.set_camera_state(old_camera)

    pil_img = Image.fromarray(img_arr)
    pil_img.info["dpi"] = (dpi, dpi)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def save_screenshot(
    scene: VizScene,
    filepath: str | Path,
    *,
    scale: int = DEFAULT_SCALE,
    dpi: int = DEFAULT_DPI,
    transparent_background: bool = False,
) -> Path:
    """Save a 300-DPI PNG screenshot to disk.

    Parameters
    ----------
    scene : VizScene
        The scene to capture.
    filepath : str or Path
        Output file path (should end with ``.png``).
    scale : int
        Pixel multiplier.
    dpi : int
        Target DPI.
    transparent_background : bool
        Transparent background flag.

    Returns
    -------
    Path
        The saved file path.
    """
    data = screenshot_bytes(
        scene,
        scale=scale,
        dpi=dpi,
        transparent_background=transparent_background,
    )
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out
