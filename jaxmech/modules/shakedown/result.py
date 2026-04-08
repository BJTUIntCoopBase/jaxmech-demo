"""Result container for shakedown analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class ShakedownResult:
    """Structured result of a shakedown analysis."""

    alpha: float = np.nan
    status: str = "unknown"
    solver: str = ""
    residual_stress: Optional[np.ndarray] = None
    yield_ratio: Optional[np.ndarray] = None
    summary_mat_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Shakedown Analysis [{self.status.upper()}]",
            f"  solver: {self.solver}",
            f"  alpha:  {self.alpha:.12f}",
        ]
        if self.metadata.get("solve_time"):
            lines.append(f"  time:   {self.metadata['solve_time']:.2f}s")
        if self.summary_mat_path:
            lines.append(f"  output: {self.summary_mat_path}")
        return "\n".join(lines)
