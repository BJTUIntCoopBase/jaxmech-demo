"""Create or sync Config/env.cfg from Config/env.template.cfg."""

from __future__ import annotations

from jaxmech.env.env import ensure_env_cfg_exists


def main() -> None:
    env_path = ensure_env_cfg_exists()
    if env_path is None:
        raise SystemExit("Config/env.template.cfg not found.")
    print(env_path)


if __name__ == "__main__":
    main()
