"""Route checks for the Version 5A richer fixed-map layouts.

These tests prove the maps have multiple valid success paths before PPO training:
- non-jump detours around the obstacle field,
- and a shorter jump-based shortcut through the middle.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.train_ppo import build_env_from_config, load_config


TOP_ROUTE_ACTIONS = [
    0,
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
    2,
    0,
    0,
    0,
]

BOTTOM_ROUTE_ACTIONS = [
    0,
    0,
    2,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
]

JUMP_SHORTCUT_ACTIONS = [
    0,
    0,
    0,
    3,
    0,
    0,
    0,
    3,
    0,
]


def run_route(config_path: str, actions: list[int]) -> tuple[bool, bool, tuple[int, int]]:
    """Execute a fixed route and return success, truncation, and final position."""
    config = load_config(Path(config_path))
    env = build_env_from_config(config)
    env.reset()
    info = {"success": False}
    truncated = False

    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return bool(info["success"]), bool(truncated), env.position


class WarehouseNavigationV5AMapTests(unittest.TestCase):
    """Manual route validation for the richer fixed Version 5A maps."""

    def test_map1_has_two_non_jump_success_routes(self) -> None:
        success, truncated, final_position = run_route("configs/ppo_config_v5a_map1.yaml", TOP_ROUTE_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (12, 6))

        success, truncated, final_position = run_route("configs/ppo_config_v5a_map1.yaml", BOTTOM_ROUTE_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (12, 6))

    def test_map1_has_short_jump_shortcut(self) -> None:
        success, truncated, final_position = run_route("configs/ppo_config_v5a_map1.yaml", JUMP_SHORTCUT_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (12, 6))

    def test_map2_has_safe_detour_and_jump_shortcut(self) -> None:
        success, truncated, final_position = run_route("configs/ppo_config_v5a_map2.yaml", TOP_ROUTE_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (12, 6))

        success, truncated, final_position = run_route("configs/ppo_config_v5a_map2.yaml", JUMP_SHORTCUT_ACTIONS)
        self.assertTrue(success)
        self.assertFalse(truncated)
        self.assertEqual(final_position, (12, 6))


if __name__ == "__main__":
    unittest.main()
