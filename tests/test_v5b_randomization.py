"""Checks for the first generalization-oriented map variants.

These tests make sure the environment can now:
- randomize the start while keeping the target fixed,
- randomize the target while keeping the start fixed,
- and keep those sampled positions inside the intended candidate sets.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.train_ppo import build_env_from_config, load_config


class WarehouseNavigationV5BRandomizationTests(unittest.TestCase):
    """Validate the randomized reset behavior for the next map variants."""

    def test_random_starts_use_all_configured_candidates(self) -> None:
        env = build_env_from_config(load_config(Path("configs/ppo_config_v5b_random_starts.yaml")))
        seen_starts: set[tuple[int, int]] = set()
        seen_targets: set[tuple[int, int]] = set()

        for seed in range(40):
            env.reset(seed=seed)
            seen_starts.add(env.start_pos)
            seen_targets.add(env.target_pos)

        self.assertEqual(seen_starts, {(1, 3), (1, 6), (1, 9)})
        self.assertEqual(seen_targets, {(12, 6)})

    def test_random_targets_use_all_configured_candidates(self) -> None:
        env = build_env_from_config(load_config(Path("configs/ppo_config_v5c_random_targets.yaml")))
        seen_starts: set[tuple[int, int]] = set()
        seen_targets: set[tuple[int, int]] = set()

        for seed in range(40):
            env.reset(seed=seed)
            seen_starts.add(env.start_pos)
            seen_targets.add(env.target_pos)

        self.assertEqual(seen_starts, {(1, 6)})
        self.assertEqual(seen_targets, {(12, 3), (12, 6), (12, 9)})


if __name__ == "__main__":
    unittest.main()
