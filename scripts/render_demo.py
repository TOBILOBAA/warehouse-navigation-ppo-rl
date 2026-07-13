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

import numpy as np
from stable_baselines3 import PPO

from env.warehouse_navigation_env import ACTION_NAMES
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
    parser.add_argument(
        "--show-policy-details",
        action="store_true",
        help="Print action probabilities and the chosen action at each frame.",
    )
    parser.add_argument(
        "--show-observation",
        action="store_true",
        help="Print the numeric observation vector at each frame.",
    )
    return parser.parse_args()


def clear_terminal() -> None:
    """Clear the terminal between frames for a simple animation effect."""
    print("\033[2J\033[H", end="")


def get_action_probabilities(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Return PPO's current action probabilities for a single observation."""
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(obs_tensor)
    return distribution.distribution.probs.detach().cpu().numpy()[0]


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
        last_action_name = None
        last_reward = None
        last_info = None

        while not done and not truncated:
            clear_terminal()
            print(f"Episode {episode_index + 1}/{args.episodes}")
            print(f"Frame {frame_index}")
            print(env.render())
            print(f"total_reward={total_reward:.3f}")
            print(f"deterministic={deterministic}")
            if args.show_observation:
                print(f"observation={np.array2string(observation, precision=3)}")
            if last_action_name is not None:
                print(f"last_action={last_action_name} last_reward={last_reward:.3f}")
                print(
                    "last_events="
                    f"success={last_info['success']} "
                    f"collision={last_info['collision']} "
                    f"blocked={last_info['blocked']} "
                    f"failed_jump={last_info['failed_jump']}"
                )

            action_probabilities = None
            if args.show_policy_details:
                action_probabilities = get_action_probabilities(model, observation)
                formatted_probs = ", ".join(
                    f"{ACTION_NAMES[action_index]}={probability:.3f}"
                    for action_index, probability in enumerate(action_probabilities)
                )
                print(f"policy_probs: {formatted_probs}")

            action, _state = model.predict(observation, deterministic=deterministic)
            action_name = ACTION_NAMES[int(action)]
            if args.show_policy_details:
                print(f"chosen_action={action_name}")
            observation, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
            last_action_name = action_name
            last_reward = reward
            last_info = info
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
