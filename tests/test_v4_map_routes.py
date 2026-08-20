"""Route checks for the harder Version 4 grid layout.

These tests prove the map has more than one successful path to the target,
which is exactly what we want before we train PPO on it.
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
    2,
    3,
    0,
    1,
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
    1,
    3,
    0,
    2,
    0,
    0,
]


class WarehouseNavigationV4RouteTests(unittest.TestCase):
    """Manual route validation for the harder multi-path grid."""

    def setUp(self) -> None:
        self.config = load_config(Path("configs/ppo_config_v4.yaml"))

    def test_top_route_reaches_target(self) -> None:
        env = build_env_from_config(self.config)
        env.reset()

        for action in TOP_ROUTE_ACTIONS:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(info["success"])
        self.assertFalse(truncated)
        self.assertEqual(env.position, (10, 6))

    def test_bottom_route_reaches_target(self) -> None:
        env = build_env_from_config(self.config)
        env.reset()

        for action in BOTTOM_ROUTE_ACTIONS:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        self.assertTrue(info["success"])
        self.assertFalse(truncated)
        self.assertEqual(env.position, (10, 6))


if __name__ == "__main__":
    unittest.main()
