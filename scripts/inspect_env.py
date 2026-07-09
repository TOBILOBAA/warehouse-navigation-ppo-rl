"""Manual environment walkthrough for human inspection.

This script does not train PPO.
It simply plays a fixed action sequence so we can verify:
- the map renders correctly,
- blocked moves behave correctly,
- jump works,
- rewards and termination look sensible.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow the script to import the local env package when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import WarehouseNavigationEnv


# A short hand-written route that demonstrates blocked forward movement,
# a successful jump, and eventually reaching the target.
DEMO_ACTIONS = [
    0,  # forward
    0,  # forward
    0,  # forward
    3,  # jump
    0,  # forward
    0,  # forward
    0,  # forward -> target
]


def main() -> None:
    """Run one deterministic episode and print what happens at each step."""
    env = WarehouseNavigationEnv()
    observation, info = env.reset()

    print("Initial observation:")
    print(observation)
    print("Initial info:")
    print(info)
    print()
    print(env.render())
    print()

    for step_index, action in enumerate(DEMO_ACTIONS, start=1):
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_index}: action={action}")
        print(f"reward={reward:.3f} terminated={terminated} truncated={truncated}")
        print(info)
        print(env.render())
        print()

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
