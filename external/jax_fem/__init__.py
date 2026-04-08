# Modified from upstream JAX-FEM: https://github.com/deepmodeling/jax-fem
# Upstream license: GPL-3.0
# Local jaxmech changes keep this vendored copy compatible with this repository.

try:
    import basix  # type: ignore
except ModuleNotFoundError:
    basix = None
from pyfiglet import Figlet

f = Figlet(font='starwars')
print(f.renderText('JAX - FEM'))

from .logger_setup import setup_logger
# LOGGING
logger = setup_logger(__name__)

# TODO: Be automatic
# __version__ = "0.0.11"
