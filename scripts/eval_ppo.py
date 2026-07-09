"""Evaluate a saved PPO checkpoint on the warehouse navigation environment.

This script answers the question:
"Given a trained policy, how well does it actually navigate the warehouse?"

It loads:
- the same environment config used for training,
- a saved PPO checkpoint,
- then runs a fixed number of evaluation episodes and summarizes the results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stable Baselines3 imports matplotlib through its logging stack, so we point
# matplotlib's config/cache at a writable project-local directory before import.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from stable_baselines3 import PPO

from scripts.train_ppo import build_env_from_config, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line options for PPO evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate PPO on WarehouseNavigationEnv")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ppo_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "smoke_test.zip",
        help="Path to the saved PPO checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for how many evaluation episodes to run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="smoke_test_eval",
        help="Name used for the saved evaluation summary.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of using deterministic policy output.",
    )
    parser.add_argument(
        "--render-first-episode",
        action="store_true",
        help="Print the environment state at each step for the first evaluation episode.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path for the evaluation JSON report.",
    )
    return parser.parse_args()


def compute_summary(episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode results into high-level evaluation metrics."""
    num_episodes = len(episode_rows)
    successes = [row for row in episode_rows if row["success"]]
    total_collision_events = sum(row["collision_events"] for row in episode_rows)
    total_jump_counts = sum(row["jump_count"] for row in episode_rows)
    avg_reward = sum(row["episode_reward"] for row in episode_rows) / num_episodes
    success_rate = len(successes) / num_episodes
    avg_steps_success = (
        sum(row["steps"] for row in successes) / len(successes)
        if successes
        else None
    )
    timeouts = [row for row in episode_rows if row["truncated"]]
    stuck_episodes = [
        row for row in episode_rows
        if (not row["success"]) and row["collision_events"] == 0 and row["jump_count"] == 0
    ]

    return {
        "episodes": num_episodes,
        "success_rate": success_rate,
        "average_episode_reward": avg_reward,
        "average_steps_per_successful_episode": avg_steps_success,
        "collision_rate": total_collision_events / num_episodes,
        "average_jumps_per_episode": total_jump_counts / num_episodes,
        "timeout_rate": len(timeouts) / num_episodes,
        "stuck_rate": len(stuck_episodes) / num_episodes,
    }


def main() -> None:
    """Run the saved PPO policy for a fixed number of episodes and summarize results."""
    args = parse_args()
    config = load_config(args.config)
    env = build_env_from_config(config)
    model = PPO.load(str(args.model_path))
    eval_config = dict(config.get("eval", {}))
    num_episodes = args.episodes if args.episodes is not None else int(eval_config.get("episodes", 20))
    deterministic = not args.stochastic if args.stochastic else bool(eval_config.get("deterministic", True))

    episode_rows: list[dict[str, Any]] = []

    for episode_index in range(num_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        collision_events = 0
        rendered_frames: list[str] = []

        if args.render_first_episode and episode_index == 0:
            rendered_frames.append(env.render())

        while not done and not truncated:
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, done, truncated, info = env.step(int(action))
            episode_reward += reward
            collision_events += int(info["collision"])
            if args.render_first_episode and episode_index == 0:
                rendered_frames.append(env.render())

        row = {
            "episode_index": episode_index,
            "success": bool(info["success"]),
            "steps": env.step_count,
            "episode_reward": float(episode_reward),
            "collision_events": collision_events,
            "jump_count": int(info["jump_count"]),
            "final_position": list(info["position"]),
            "terminated": bool(done),
            "truncated": bool(truncated),
        }
        if args.render_first_episode and episode_index == 0:
            row["render_trace"] = rendered_frames
        episode_rows.append(row)

    summary = compute_summary(episode_rows)

    eval_dir = PROJECT_ROOT / "results" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_path if args.output_path is not None else eval_dir / f"{args.run_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "episodes": episode_rows}, indent=2),
        encoding="utf-8",
    )

    print("Evaluation summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    if args.render_first_episode and episode_rows:
        print("\nFirst episode trace:")
        for step_index, frame in enumerate(episode_rows[0].get("render_trace", [])):
            print(f"\nFrame {step_index}")
            print(frame)
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
