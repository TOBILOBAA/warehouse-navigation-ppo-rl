"""Route checks for the balanced fixed-endpoint map.

The goal here is not just "can the robot reach the target?"
It is "does the map offer different navigation styles without making jump
the automatic best answer every time?"
"""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.train_ppo import build_env_from_config, load_config


NO_JUMP_ROUTE_ACTIONS = [
    0,
    1,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
]

BALANCED_JUMP_ROUTE_ACTIONS = [
    0,
    1,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    0,
    0,
    3,
    0,
]


def run_route(actions: list[int]) -> tuple[bool, bool, tuple[int, int], int]:
    """Execute a fixed route on the balanced map and return the outcome."""
    config = load_config(Path("configs/ppo_config_v5d_balanced_routes.yaml"))
    env = build_env_from_config(config)
    env.reset(seed=0)
    info = {"success": False}
    truncated = False

    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return bool(info["success"]), bool(truncated), env.position, env.jump_count


class WarehouseNavigationV5DBalancedRoutesTests(unittest.TestCase):
    """Validate that the balanced fixed map has meaningful route options."""

    def test_balanced_map_has_successful_no_jump_route(self) -> None:
        success, truncated, final_position, jump_count = run_route(NO_JUMP_ROUTE_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (14, 6))
        self.assertEqual(jump_count, 0)

    def test_balanced_map_has_successful_jump_route(self) -> None:
        success, truncated, final_position, jump_count = run_route(BALANCED_JUMP_ROUTE_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (14, 6))
        self.assertEqual(jump_count, 1)


if __name__ == "__main__":
    unittest.main()
