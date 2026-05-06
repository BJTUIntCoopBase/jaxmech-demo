"""jaxmech.modules.visualization — lightweight web-based field visualization."""

from jaxmech.modules.visualization.mat_reader import VizData, inspect_mat_fields, load_viz_data
from jaxmech.modules.visualization.mesh_builder import build_grid, gauss_to_nodal
from jaxmech.modules.visualization.scene import VizScene
from jaxmech.modules.visualization.session import VizSession
from jaxmech.modules.visualization.screenshot import screenshot_bytes, save_screenshot

__all__ = [
    "VizData",
    "inspect_mat_fields",
    "load_viz_data",
    "build_grid",
    "gauss_to_nodal",
    "VizScene",
    "VizSession",
    "screenshot_bytes",
    "save_screenshot",
]
