"""Validate the custom environment against Gymnasium's environment checker.

This is our first formal "does the API look right?" script.
It does not train PPO; it only checks whether the environment behaves like a
proper Gymnasium environment.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import WarehouseNavigationEnv


def main() -> None:
    try:
        from gymnasium.utils.env_checker import check_env
    except ImportError:
        print("Gymnasium is not installed yet.")
        print("Install dependencies first, then rerun: python3 scripts/check_env.py")
        return

    env = WarehouseNavigationEnv(render_mode="ansi")
    check_env(env)
    print("Environment check passed.")


if __name__ == "__main__":
    main()
