"""Render a saved PPO policy step by step in the terminal.

This is the "watch the robot move" script.
It loads a trained checkpoint, runs one or more episodes, and prints the grid
frame by frame so we can visually inspect what the policy is doing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from stable_baselines3 import PPO

from scripts.train_ppo import build_env_from_config, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line options for terminal rendering of a saved policy."""
    parser = argparse.ArgumentParser(description="Render a trained PPO policy in the terminal")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ppo_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "baseline_v1.zip",
        help="Path to the saved PPO checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to render.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between frames.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of using deterministic policy output.",
    )
    return parser.parse_args()


def clear_terminal() -> None:
    """Clear the terminal between frames for a simple animation effect."""
    print("\033[2J\033[H", end="")


def main() -> None:
    """Run one or more rendered episodes using a saved PPO checkpoint."""
    args = parse_args()
    config = load_config(args.config)
    env = build_env_from_config(config)
    model = PPO.load(str(args.model_path))
    deterministic = not args.stochastic

    for episode_index in range(args.episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        frame_index = 0

        while not done and not truncated:
            clear_terminal()
            print(f"Episode {episode_index + 1}/{args.episodes}")
            print(f"Frame {frame_index}")
            print(env.render())
            print(f"total_reward={total_reward:.3f}")
            print(f"deterministic={deterministic}")

            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            frame_index += 1
            time.sleep(args.delay)

        clear_terminal()
        print(f"Episode {episode_index + 1}/{args.episodes} complete")
        print(env.render())
        print(f"success={info['success']} truncated={truncated}")
        print(f"steps={env.step_count} total_reward={total_reward:.3f}")
        if episode_index < args.episodes - 1:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
