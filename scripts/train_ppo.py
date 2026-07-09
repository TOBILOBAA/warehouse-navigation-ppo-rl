"""Train a PPO baseline on the grid-world warehouse navigation environment.

This script is the first full RL loop in the project:
- load config,
- build the environment,
- create a PPO model,
- train it,
- save logs and the trained checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stable Baselines3 imports matplotlib through its logging stack, so we point
# matplotlib's config/cache at a writable project-local directory.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env import WarehouseNavigationEnv
from env.warehouse_navigation_env import RewardConfig
from scripts.run_utils import append_jsonl, human_duration, iso_now, load_monitor_summary, write_json


def tensorboard_available() -> bool:
    """Return True when tensorboard is installed in the current environment."""
    return importlib.util.find_spec("tensorboard") is not None


def parse_args() -> argparse.Namespace:
    """Parse command-line options for baseline PPO training."""
    parser = argparse.ArgumentParser(description="Train PPO on WarehouseNavigationEnv")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ppo_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional override for total training timesteps.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ppo_baseline",
        help="Name used for the output folder and saved checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PPO and the environment.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Optional override for PPO rollout length during quick smoke tests.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration for the environment, rewards, and PPO."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_env_from_config(config: dict[str, Any]) -> WarehouseNavigationEnv:
    """Create the custom Gymnasium environment from the YAML config."""
    env_config = dict(config["env"])
    reward_config = RewardConfig(**config["reward"])

    # YAML stores coordinate pairs as lists; the environment expects tuples.
    env_config["start_pos"] = tuple(env_config["start_pos"])
    env_config["target_pos"] = tuple(env_config["target_pos"])

    return WarehouseNavigationEnv(
        reward_config=reward_config,
        render_mode="ansi",
        **env_config,
    )


def main() -> None:
    """Run the full PPO baseline training flow."""
    args = parse_args()
    config = load_config(args.config)

    results_root = PROJECT_ROOT / "results"
    run_dir = results_root / "runs" / args.run_name
    logs_dir = run_dir / "logs"
    models_dir = results_root / "models"
    mpl_config_dir = PROJECT_ROOT / ".mplconfig"
    monitor_path = logs_dir / "monitor.csv"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    training_metadata_path = run_dir / "training_metadata.json"
    training_summary_path = run_dir / "training_summary.json"
    registry_path = results_root / "run_registry.jsonl"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    env = build_env_from_config(config)
    env.reset(seed=args.seed)
    monitored_env = Monitor(env, filename=str(monitor_path))

    ppo_config = dict(config["ppo"])
    config_total_timesteps = int(ppo_config.pop("total_timesteps"))
    total_timesteps = args.timesteps if args.timesteps is not None else config_total_timesteps
    policy = ppo_config.pop("policy")
    if args.n_steps is not None:
        ppo_config["n_steps"] = args.n_steps
    tb_log_dir = str(logs_dir) if tensorboard_available() else None
    if tb_log_dir is None:
        print("TensorBoard is not installed; training will continue without tensorboard logs.")

    effective_config = {
        "env": config["env"],
        "reward": config["reward"],
        "ppo": {
            **config["ppo"],
            "policy": policy,
            "total_timesteps": total_timesteps,
            **({"n_steps": args.n_steps} if args.n_steps is not None else {}),
        },
        "eval": config.get("eval", {}),
        "run": {
            "run_name": args.run_name,
            "seed": args.seed,
            "tensorboard_enabled": tb_log_dir is not None,
        },
    }
    config_snapshot_path.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")

    # PPO consumes observations from the environment and learns a policy
    # that maps those observations to actions over many episodes.
    model = PPO(
        policy,
        monitored_env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        seed=args.seed,
        **ppo_config,
    )

    print(f"Training PPO for {total_timesteps} timesteps...")
    started_at = iso_now()
    start_perf = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    duration_seconds = time.perf_counter() - start_perf
    completed_at = iso_now()

    model_path = models_dir / f"{args.run_name}.zip"
    model.save(str(model_path))
    print(f"Saved trained model to: {model_path}")
    print(f"Monitor outputs live under: {logs_dir}")
    if tb_log_dir is not None:
        print(f"TensorBoard logs live under: {logs_dir}")

    training_summary = load_monitor_summary(monitor_path)
    training_summary.update(
        {
            "run_name": args.run_name,
            "timesteps_requested": total_timesteps,
            "duration_seconds": duration_seconds,
            "duration_human": human_duration(duration_seconds),
            "started_at": started_at,
            "completed_at": completed_at,
        }
    )
    write_json(training_summary_path, training_summary)

    deterministic_eval_path = run_dir / "evaluation_deterministic.json"
    stochastic_eval_path = run_dir / "evaluation_stochastic.json"

    for output_path, extra_args in [
        (deterministic_eval_path, []),
        (stochastic_eval_path, ["--stochastic"]),
    ]:
        eval_run_name = output_path.stem
        eval_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_ppo.py"),
            "--config",
            str(args.config),
            "--model-path",
            str(model_path),
            "--run-name",
            eval_run_name,
            "--output-path",
            str(output_path),
            *extra_args,
        ]
        subprocess.run(eval_command, check=True, cwd=str(PROJECT_ROOT))

    deterministic_eval = json.loads(deterministic_eval_path.read_text(encoding="utf-8"))
    stochastic_eval = json.loads(stochastic_eval_path.read_text(encoding="utf-8"))

    training_metadata = {
        "run_name": args.run_name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "duration_human": human_duration(duration_seconds),
        "seed": args.seed,
        "config_path": args.config,
        "config_snapshot_path": config_snapshot_path,
        "model_path": model_path,
        "monitor_path": monitor_path,
        "logs_dir": logs_dir,
        "training_summary_path": training_summary_path,
        "deterministic_evaluation_path": deterministic_eval_path,
        "stochastic_evaluation_path": stochastic_eval_path,
        "effective_config": effective_config,
        "training_summary": training_summary,
        "deterministic_evaluation_summary": deterministic_eval["summary"],
        "stochastic_evaluation_summary": stochastic_eval["summary"],
    }
    write_json(training_metadata_path, training_metadata)
    append_jsonl(registry_path, training_metadata)
    print(f"Saved run metadata to: {training_metadata_path}")


if __name__ == "__main__":
    main()
