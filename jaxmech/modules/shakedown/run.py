"""CLI entry point for cfg-driven shakedown analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxmech.modules.shakedown import run_shakedown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the .cfg file.")
    parser.add_argument(
        "--step",
        choices=("all", "prepare", "collect"),
        default="all",
        help="Execution step. The demo CVXPY workflow supports 'all'.",
    )
    args = parser.parse_args()

    result = run_shakedown(Path(args.config), step=args.step)
    print(result.summary())
    if result.summary_mat_path:
        print(f"__ARTIFACT__|mat|{result.summary_mat_path}|Shakedown result MAT")


if __name__ == "__main__":
    main()
