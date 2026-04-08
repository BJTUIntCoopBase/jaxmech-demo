"""Compatibility wrapper for the build-category tool entry point."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from jaxmech.tools.build.build_model_mat import main, run_jax_elastic

__all__ = ["main", "run_jax_elastic"]


if __name__ == "__main__":
    main()
