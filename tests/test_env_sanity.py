"""Sanity checks for the grid-world navigation environment.

These tests are not PPO training tests.
They are environment-behavior tests that answer:
- does movement work,
- do penalties fire when expected,
- do turning and observations update correctly,
- and do episodes end only when the MDP says they should?
"""

from __future__ import annotations

import unittest

import numpy as np

from env import WarehouseNavigationEnv


class WarehouseNavigationEnvSanityTests(unittest.TestCase):
    """Focused behavior checks for the Version 0 environment."""

    def test_forward_move_advances_robot_and_gives_positive_progress_reward(self) -> None:
        env = WarehouseNavigationEnv()
        env.reset()

        observation, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(env.position, (2, 5))
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["blocked"])
        self.assertFalse(info["collision"])

    def test_blocked_forward_by_low_obstacle_is_non_terminal(self) -> None:
        env = WarehouseNavigationEnv()
        env.reset()
        env.position = (3, 5)
        env.heading = 1  # east, obstacle at (4, 5)

        _, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(env.position, (3, 5))
        self.assertAlmostEqual(reward, -0.21, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["blocked"])
        self.assertFalse(info["collision"])

    def test_wall_collision_attempt_is_non_terminal_and_stays_in_place(self) -> None:
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[1, 2] = 1  # wall directly in front of start
        env = WarehouseNavigationEnv(
            width=3,
            height=3,
            grid=grid,
            start_pos=(1, 1),
            start_heading=1,
            target_pos=(0, 0),
        )
        env.reset()

        _, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(env.position, (1, 1))
        self.assertAlmostEqual(reward, -0.51, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertFalse(info["blocked"])

    def test_failed_jump_is_non_terminal_and_penalized(self) -> None:
        env = WarehouseNavigationEnv()
        env.reset()

        _, reward, terminated, truncated, info = env.step(3)

        self.assertEqual(env.position, (1, 5))
        self.assertAlmostEqual(reward, -0.31, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["failed_jump"])

    def test_successful_jump_moves_two_cells_forward(self) -> None:
        env = WarehouseNavigationEnv()
        env.reset()
        env.position = (3, 5)
        env.heading = 1  # east, low obstacle ahead at (4, 5)

        _, reward, terminated, truncated, info = env.step(3)

        self.assertEqual(env.position, (5, 5))
        self.assertGreater(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["failed_jump"])
        self.assertEqual(info["jump_count"], 1)

    def test_turning_changes_heading_and_observation(self) -> None:
        grid = np.zeros((3, 3), dtype=np.int8)
        grid[1, 2] = 1  # east cell is a wall
        grid[0, 1] = 2  # north cell is a low obstacle
        env = WarehouseNavigationEnv(
            width=3,
            height=3,
            grid=grid,
            start_pos=(1, 1),
            start_heading=1,
            target_pos=(2, 2),
        )
        observation_before, _ = env.reset()

        observation_after, reward, terminated, truncated, info = env.step(1)

        self.assertEqual(env.heading, 0)  # now facing north
        self.assertAlmostEqual(reward, -0.01, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(observation_before[7], observation_after[7])  # front sensor changed
        self.assertEqual(info["heading"], "north")

    def test_episode_truncates_when_max_steps_is_reached(self) -> None:
        env = WarehouseNavigationEnv(max_steps=1)
        env.reset()

        _, _, terminated, truncated, _ = env.step(1)

        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_reaching_target_terminates_episode(self) -> None:
        env = WarehouseNavigationEnv()
        env.reset()
        env.position = (7, 5)
        env.heading = 1  # east, target at (8, 5)

        _, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(env.position, (8, 5))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertGreater(reward, 10.0)


if __name__ == "__main__":
    unittest.main()
