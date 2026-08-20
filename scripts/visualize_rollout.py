"""Create visual rollout frames for a trained PPO policy.

This script is meant for learning and debugging:
- it renders the warehouse as a colored grid,
- overlays the robot trajectory,
- annotates each frame with the chosen action and reward,
- and saves episode-by-episode deterministic vs stochastic comparisons.
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

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from stable_baselines3 import PPO

from env.warehouse_navigation_env import ACTION_NAMES, FREE, HEADING_NAMES, HEADING_VECTORS, LOW_OBSTACLE, WALL
from scripts.train_ppo import build_env_from_config, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line options for rollout visualization."""
    parser = argparse.ArgumentParser(description="Visualize a PPO rollout on WarehouseNavigationEnv")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ppo_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "results" / "models" / "baseline_v2.zip",
        help="Path to the saved PPO checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "visuals",
        help="Directory where rollout frames and summaries will be saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline_v2",
        help="Folder prefix used under the output directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episode report folders to generate.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=1,
        help="Episode index to start numbering from when saving reports.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on how many rollout steps to visualize.",
    )
    return parser.parse_args()


def get_action_probabilities(model: PPO, observation: np.ndarray) -> np.ndarray:
    """Return PPO's action probabilities for one observation."""
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(obs_tensor)
    return distribution.distribution.probs.detach().cpu().numpy()[0]


def run_rollout(
    model: PPO,
    env: Any,
    *,
    deterministic: bool,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    """Run a single episode and collect rich per-step debug data."""
    observation, info = env.reset()
    frames: list[dict[str, Any]] = [
        {
            "frame_index": 0,
            "position": env.position,
            "heading": env.heading,
            "action_name": None,
            "reward": None,
            "info": info,
            "observation": observation.copy(),
            "policy_probs": get_action_probabilities(model, observation),
        }
    ]

    done = False
    truncated = False

    while not done and not truncated:
        if max_steps is not None and env.step_count >= max_steps:
            break

        policy_probs = get_action_probabilities(model, observation)
        action, _state = model.predict(observation, deterministic=deterministic)
        action_index = int(action)
        observation, reward, done, truncated, info = env.step(action_index)
        frames.append(
            {
                "frame_index": len(frames),
                "position": env.position,
                "heading": env.heading,
                "action_name": ACTION_NAMES[action_index],
                "reward": float(reward),
                "info": info,
                "observation": observation.copy(),
                "policy_probs": policy_probs,
            }
        )

    return frames


def draw_grid(ax: plt.Axes, env: Any, path_positions: list[tuple[int, int]], heading: int) -> None:
    """Draw the warehouse map and the robot trajectory on a matplotlib axis."""
    ax.set_aspect("equal")
    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    ax.grid(color="#d5d8dc", linewidth=0.8)
    ax.set_facecolor("#f7f4ec")

    color_by_cell = {
        FREE: "#f8f6f1",
        WALL: "#3f3f46",
        LOW_OBSTACLE: "#d97706",
    }

    for y in range(env.height):
        for x in range(env.width):
            cell_value = int(env.grid[y, x])
            rect = Rectangle(
                (x, y),
                1,
                1,
                facecolor=color_by_cell.get(cell_value, "#f8f6f1"),
                edgecolor="#d5d8dc",
                linewidth=0.8,
            )
            ax.add_patch(rect)

    start_x, start_y = env.start_pos
    target_x, target_y = env.target_pos
    ax.add_patch(Rectangle((target_x, target_y), 1, 1, facecolor="#16a34a", alpha=0.75, edgecolor="none"))
    ax.text(start_x + 0.5, start_y + 0.52, "S", ha="center", va="center", fontsize=10, weight="bold", color="#1f2937")
    ax.text(target_x + 0.5, target_y + 0.52, "T", ha="center", va="center", fontsize=10, weight="bold", color="white")

    if path_positions:
        xs = [x + 0.5 for x, _ in path_positions]
        ys = [y + 0.5 for _, y in path_positions]
        ax.plot(xs, ys, color="#2563eb", linewidth=2.2, alpha=0.9)
        for step_index, (x, y) in enumerate(path_positions):
            ax.text(x + 0.5, y + 0.18, str(step_index), ha="center", va="center", fontsize=7, color="#0f172a")

        robot_x, robot_y = path_positions[-1]
        ax.add_patch(Circle((robot_x + 0.5, robot_y + 0.5), 0.22, color="#dc2626", zorder=4))
        dx, dy = HEADING_VECTORS[heading]
        ax.arrow(
            robot_x + 0.5,
            robot_y + 0.5,
            dx * 0.22,
            dy * 0.22,
            width=0.03,
            head_width=0.14,
            head_length=0.12,
            color="white",
            length_includes_head=True,
            zorder=5,
        )

    ax.set_title("Warehouse Layout and Robot Path", fontsize=12)


def format_policy_probabilities(policy_probs: np.ndarray) -> list[str]:
    """Format PPO probabilities as readable labels."""
    return [
        f"{ACTION_NAMES[action_index]}: {probability:.3f}"
        for action_index, probability in enumerate(policy_probs)
    ]


def save_frame_visuals(
    env: Any,
    rollout: list[dict[str, Any]],
    output_dir: Path,
    mode_name: str,
) -> None:
    """Save one PNG per frame plus a summary JSON for the rollout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path_positions: list[tuple[int, int]] = []

    for frame in rollout:
        path_positions.append(tuple(frame["position"]))
        fig, (ax_map, ax_text) = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
        )
        draw_grid(ax_map, env, path_positions, frame["heading"])

        ax_text.axis("off")
        info = frame["info"]
        status_lines = [
            f"Mode: {mode_name}",
            f"Frame: {frame['frame_index']}",
            f"Position: {frame['position']}",
            f"Heading: {HEADING_NAMES[frame['heading']]}",
            f"Chosen action: {frame['action_name'] or 'start'}",
            f"Reward from action: {frame['reward'] if frame['reward'] is not None else 'n/a'}",
            f"Distance to target: {info['distance_to_target']:.3f}",
            f"Success: {info['success']}",
            f"Collision: {info['collision']}",
            f"Blocked: {info['blocked']}",
            f"Failed jump: {info['failed_jump']}",
            f"Jumps so far: {info['jump_count']}",
            f"Steps used: {frame['frame_index']}",
        ]
        policy_lines = ["Policy action probabilities:"] + format_policy_probabilities(frame["policy_probs"])
        observation_lines = [
            "Observation vector:",
            np.array2string(frame["observation"], precision=3, separator=", "),
        ]
        ax_text.text(
            0.0,
            1.0,
            "\n".join(status_lines + [""] + policy_lines + [""] + observation_lines),
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        fig.suptitle(f"{mode_name.title()} Rollout", fontsize=14, weight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{frame['frame_index']:03d}.png", dpi=160)
        plt.close(fig)

    summary_rows = [
        {
            "frame_index": frame["frame_index"],
            "position": list(frame["position"]),
            "heading": HEADING_NAMES[frame["heading"]],
            "action_name": frame["action_name"],
            "reward": frame["reward"],
            "info": frame["info"],
            "policy_probs": {
                ACTION_NAMES[action_index]: float(probability)
                for action_index, probability in enumerate(frame["policy_probs"])
            },
        }
        for frame in rollout
    ]
    (output_dir / "rollout_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def build_rollout_summary(rollout: list[dict[str, Any]], mode_name: str) -> dict[str, Any]:
    """Create a compact summary for one finished rollout."""
    final_frame = rollout[-1]
    rewards = [frame["reward"] for frame in rollout if frame["reward"] is not None]
    return {
        "mode": mode_name,
        "frame_count": len(rollout),
        "steps": len(rollout) - 1,
        "success": bool(final_frame["info"]["success"]),
        "final_position": list(final_frame["position"]),
        "final_heading": HEADING_NAMES[final_frame["heading"]],
        "total_reward": float(sum(rewards)),
        "jump_count": int(final_frame["info"]["jump_count"]),
        "distance_to_target": float(final_frame["info"]["distance_to_target"]),
    }


def save_comparison_figure(
    det_env: Any,
    stoch_env: Any,
    deterministic_rollout: list[dict[str, Any]],
    stochastic_rollout: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a single side-by-side path comparison image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    det_path = [tuple(frame["position"]) for frame in deterministic_rollout]
    stoch_path = [tuple(frame["position"]) for frame in stochastic_rollout]

    draw_grid(axes[0], det_env, det_path, deterministic_rollout[-1]["heading"])
    axes[0].set_title(
        "Deterministic Path\n"
        f"steps={len(deterministic_rollout) - 1}, success={deterministic_rollout[-1]['info']['success']}",
        fontsize=12,
    )

    draw_grid(axes[1], stoch_env, stoch_path, stochastic_rollout[-1]["heading"])
    axes[1].set_title(
        "Stochastic Path\n"
        f"steps={len(stochastic_rollout) - 1}, success={stochastic_rollout[-1]['info']['success']}",
        fontsize=12,
    )

    fig.suptitle("Policy Behavior Comparison", fontsize=15, weight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_episode_report(
    episode_dir: Path,
    det_env: Any,
    det_rollout: list[dict[str, Any]],
    stoch_env: Any,
    stoch_rollout: list[dict[str, Any]],
) -> None:
    """Save one episode folder containing both modes and a comparison image."""
    deterministic_dir = episode_dir / "deterministic"
    stochastic_dir = episode_dir / "stochastic"
    save_frame_visuals(det_env, det_rollout, deterministic_dir, "deterministic")
    save_frame_visuals(stoch_env, stoch_rollout, stochastic_dir, "stochastic")
    save_comparison_figure(det_env, stoch_env, det_rollout, stoch_rollout, episode_dir / "comparison.png")

    episode_summary = {
        "deterministic": build_rollout_summary(det_rollout, "deterministic"),
        "stochastic": build_rollout_summary(stoch_rollout, "stochastic"),
    }
    (episode_dir / "episode_summary.json").write_text(json.dumps(episode_summary, indent=2), encoding="utf-8")


def main() -> None:
    """Generate episode-by-episode visual rollout artifacts for one PPO checkpoint."""
    args = parse_args()
    config = load_config(args.config)
    model = PPO.load(str(args.model_path))
    run_output_dir = args.output_dir / args.run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    end_episode = args.start_episode + args.episodes
    for episode_index in range(args.start_episode, end_episode):
        det_env = build_env_from_config(config)
        stoch_env = build_env_from_config(config)
        det_rollout = run_rollout(model, det_env, deterministic=True, max_steps=args.max_steps)
        stoch_rollout = run_rollout(model, stoch_env, deterministic=False, max_steps=args.max_steps)

        episode_dir = run_output_dir / f"episode_{episode_index:03d}"
        save_episode_report(episode_dir, det_env, det_rollout, stoch_env, stoch_rollout)

        manifest.append(
            {
                "episode": episode_index,
                "folder": str(episode_dir),
                "deterministic": build_rollout_summary(det_rollout, "deterministic"),
                "stochastic": build_rollout_summary(stoch_rollout, "stochastic"),
            }
        )
        print(f"Saved episode report to: {episode_dir}")

    (run_output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved run manifest to: {run_output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
