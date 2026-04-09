"""Standalone env.cfg helper for jaxmech-demo.

Used by start_web.bat to ensure Config/env.cfg exists and to write
auto-detected values back without disturbing comments or other keys.

Usage:
    python Config/setup_env.py                    # ensure env.cfg exists
    python Config/setup_env.py --set KEY VALUE    # set a key in env.cfg
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "Config" / "env.template.cfg"
CFG = REPO_ROOT / "Config" / "env.cfg"


def ensure_cfg() -> None:
    """Create Config/env.cfg from template if it does not exist."""
    if not CFG.exists():
        CFG.parent.mkdir(parents=True, exist_ok=True)
        if TEMPLATE.exists():
            shutil.copy2(TEMPLATE, CFG)
        else:
            CFG.write_text(
                "# Machine-local environment configuration\n"
                "windows_python_exe = \n",
                encoding="utf-8",
            )


def set_key(key: str, value: str) -> None:
    """Update *key* in Config/env.cfg, preserving comments and other keys."""
    ensure_cfg()
    lines = CFG.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    found = False
    for line in lines:
        data = line.split("#", 1)[0].strip()
        if "=" in data and data.split("=", 1)[0].strip() == key:
            new_lines.append(f"{key} = {value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key} = {value}")
    CFG.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[setup_env] env.cfg updated: {key} = {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="jaxmech-demo env.cfg helper")
    parser.add_argument(
        "--set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set a key=value pair in Config/env.cfg",
    )
    args = parser.parse_args()
    ensure_cfg()
    if args.set:
        set_key(args.set[0], args.set[1])


if __name__ == "__main__":
    main()