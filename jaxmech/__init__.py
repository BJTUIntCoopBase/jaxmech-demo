"""jaxmech demo package.

This public demo contains solid elastic inc_analysis, solid CVXPY shakedown,
the Web UI, and MAT visualization.
"""

from jaxmech.env.env import ensure_env_cfg_exists

__version__ = "0.2.1-demo"

ensure_env_cfg_exists()
