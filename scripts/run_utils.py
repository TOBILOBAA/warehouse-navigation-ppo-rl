"""Small utilities for experiment tracking and run summaries."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def iso_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def human_duration(seconds: float) -> str:
    """Format elapsed time in a compact human-readable form."""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def to_jsonable(value: Any) -> Any:
    """Convert common Python objects into JSON-friendly structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one experiment record to a JSONL registry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload)) + "\n")


def load_monitor_summary(monitor_path: Path) -> dict[str, Any]:
    """Summarize SB3 monitor.csv episode results for a training run."""
    if not monitor_path.exists():
        return {
            "episode_count": 0,
            "mean_episode_reward": None,
            "mean_episode_length": None,
            "best_episode_reward": None,
            "worst_episode_reward": None,
            "final_10_mean_reward": None,
            "final_10_mean_length": None,
        }

    with monitor_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {
            "episode_count": 0,
            "mean_episode_reward": None,
            "mean_episode_length": None,
            "best_episode_reward": None,
            "worst_episode_reward": None,
            "final_10_mean_reward": None,
            "final_10_mean_length": None,
        }

    rewards = [float(row["r"]) for row in rows]
    lengths = [float(row["l"]) for row in rows]
    tail_rewards = rewards[-10:]
    tail_lengths = lengths[-10:]

    return {
        "episode_count": len(rows),
        "mean_episode_reward": sum(rewards) / len(rewards),
        "mean_episode_length": sum(lengths) / len(lengths),
        "best_episode_reward": max(rewards),
        "worst_episode_reward": min(rewards),
        "final_10_mean_reward": sum(tail_rewards) / len(tail_rewards),
        "final_10_mean_length": sum(tail_lengths) / len(tail_lengths),
    }
