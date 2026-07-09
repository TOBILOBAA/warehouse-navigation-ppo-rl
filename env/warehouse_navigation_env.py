"""First grid-world warehouse navigation environment.

This file is our Version 0 simulator:
- it stores the warehouse map,
- tracks the robot state,
- applies actions,
- builds observations for PPO,
- computes rewards,
- and renders the grid for human inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - local fallback until dependencies are installed
    gym = None
    spaces = None


# Grid cell types used inside the warehouse map.
FREE = 0
WALL = 1
LOW_OBSTACLE = 2

# Orientation and action lookup tables keep the transition logic readable.
HEADING_NAMES = ("north", "east", "south", "west")
HEADING_VECTORS = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
}
ACTION_NAMES = {
    0: "forward",
    1: "turn_left",
    2: "turn_right",
    3: "jump",
}


@dataclass(frozen=True)
class RewardConfig:
    # Keeping rewards grouped here makes it easier to tune behavior without
    # searching through the environment logic.
    target: float = 10.0
    collision: float = -0.5
    blocked_forward: float = -0.2
    failed_jump: float = -0.3
    step_penalty: float = -0.01


class WarehouseNavigationEnv(gym.Env if gym else object):
    """Grid-world warehouse navigation environment for the first PPO baseline."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        max_steps: int = 80,
        start_pos: tuple[int, int] = (1, 5),
        start_heading: int = 1,
        target_pos: tuple[int, int] = (8, 5),
        grid: np.ndarray | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = "ansi",
    ) -> None:
        # World definition: map size, start state, target state, and reward rules.
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.start_pos = start_pos
        self.start_heading = start_heading
        self.target_pos = target_pos
        self.reward_config = reward_config or RewardConfig()
        self.grid = np.array(grid, dtype=np.int8) if grid is not None else self._build_default_grid()
        self.render_mode = render_mode

        if self.render_mode not in {None, "ansi"}:
            raise ValueError(f"Unsupported render_mode {self.render_mode!r}. Expected None or 'ansi'.")

        if self.grid.shape != (self.height, self.width):
            raise ValueError(f"Grid shape {self.grid.shape} does not match {(self.height, self.width)}")

        self.position = self.start_pos
        self.heading = self.start_heading
        self.step_count = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.jump_count = 0

        if spaces is not None:
            self.action_space = spaces.Discrete(4)
            # Observation layout:
            # [position, heading, target relation, local obstacle sensors]
            low = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.5, -0.5, -0.5, -0.5], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode and return the robot's first observation."""
        if gym is not None:
            super().reset(seed=seed)

        # Episode state is reset to the fixed starting configuration.
        self.position = self.start_pos
        self.heading = self.start_heading
        self.step_count = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.jump_count = 0

        observation = self._get_observation()
        info = self._build_info(success=False, collision=False, blocked=False, failed_jump=False)
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the full RL transition tuple."""
        if action not in ACTION_NAMES:
            raise ValueError(f"Unsupported action {action}")

        # Reward shaping uses change in target distance, so we store the old value first.
        old_distance = self._distance_to_target(self.position)
        self.step_count += 1

        collision = False
        blocked = False
        failed_jump = False
        success = False
        event_reward = 0.0

        # Action application updates the hidden world state.
        if action == 0:
            collision, blocked, success, event_reward = self._handle_forward()
        elif action == 1:
            self.heading = (self.heading - 1) % 4
        elif action == 2:
            self.heading = (self.heading + 1) % 4
        elif action == 3:
            collision, failed_jump, success, event_reward = self._handle_jump()

        new_distance = self._distance_to_target(self.position)
        # Reward combines progress, time cost, and any special event on this step.
        progress_reward = old_distance - new_distance
        reward = progress_reward + self.reward_config.step_penalty + event_reward
        self.episode_reward += reward

        # terminated = task solved, truncated = action budget exhausted.
        terminated = success
        truncated = self.step_count >= self.max_steps and not terminated
        observation = self._get_observation()
        info = self._build_info(
            success=success,
            collision=collision,
            blocked=blocked,
            failed_jump=failed_jump,
        )
        return observation, float(reward), terminated, truncated, info

    def render(self) -> str:
        """Return a text view of the warehouse for human debugging."""
        if self.render_mode not in {None, "ansi"}:
            raise ValueError(f"render_mode {self.render_mode!r} is not supported by this environment.")

        rows: list[str] = []
        robot_symbols = {0: "^", 1: ">", 2: "v", 3: "<"}

        # Render uses symbols so we can inspect behavior with our eyes:
        # . free, # wall, o low obstacle, T target, arrow robot heading.
        for y in range(self.height):
            chars: list[str] = []
            for x in range(self.width):
                position = (x, y)
                if position == self.position == self.target_pos:
                    chars.append("*")
                elif position == self.position:
                    chars.append(robot_symbols[self.heading])
                elif position == self.target_pos:
                    chars.append("T")
                else:
                    cell = self._cell_value(position)
                    if cell == FREE:
                        chars.append(".")
                    elif cell == WALL:
                        chars.append("#")
                    elif cell == LOW_OBSTACLE:
                        chars.append("o")
                    else:
                        chars.append("?")
            rows.append(" ".join(chars))

        footer = (
            f"steps={self.step_count}/{self.max_steps} "
            f"heading={HEADING_NAMES[self.heading]} "
            f"position={self.position} "
            f"target={self.target_pos}"
        )
        return "\n".join(rows + [footer])

    def close(self) -> None:
        return None

    def _handle_forward(self) -> tuple[bool, bool, bool, float]:
        """Resolve a forward move based on the cell directly ahead."""
        next_pos = self._offset_position(self.position, HEADING_VECTORS[self.heading], 1)

        if not self._in_bounds(next_pos):
            self.collision_count += 1
            return True, False, False, self.reward_config.collision

        if next_pos == self.target_pos:
            self.position = next_pos
            return False, False, True, self.reward_config.target

        next_cell = self._cell_value(next_pos)
        if next_cell == FREE:
            self.position = next_pos
            return False, False, False, 0.0
        if next_cell == LOW_OBSTACLE:
            return False, True, False, self.reward_config.blocked_forward

        self.collision_count += 1
        return True, False, False, self.reward_config.collision

    def _handle_jump(self) -> tuple[bool, bool, bool, float]:
        """Resolve a jump by checking both the crossed cell and the landing cell."""
        self.jump_count += 1
        front_pos = self._offset_position(self.position, HEADING_VECTORS[self.heading], 1)
        landing_pos = self._offset_position(self.position, HEADING_VECTORS[self.heading], 2)

        if not self._in_bounds(front_pos) or not self._in_bounds(landing_pos):
            return True, True, False, self.reward_config.failed_jump

        front_cell = self._cell_value(front_pos)
        landing_cell = self._cell_value(landing_pos)

        if front_cell != LOW_OBSTACLE:
            return False, True, False, self.reward_config.failed_jump
        if landing_pos == self.target_pos:
            self.position = landing_pos
            return False, False, True, self.reward_config.target
        if landing_cell != FREE:
            return False, True, False, self.reward_config.failed_jump

        self.position = landing_pos
        return False, False, False, 0.0

    def _get_observation(self) -> np.ndarray:
        """Build the compact numeric observation that PPO receives at each step."""
        x, y = self.position
        target_dx = self.target_pos[0] - x
        target_dy = self.target_pos[1] - y
        diagonal = np.sqrt((self.width - 1) ** 2 + (self.height - 1) ** 2)
        heading_angle = (np.pi / 2.0) * self.heading

        # Local sensing is intentionally short-range in Version 0:
        # PPO only gets front, left, right, and two-cells-ahead obstacle information.
        front_vector = HEADING_VECTORS[self.heading]
        left_vector = HEADING_VECTORS[(self.heading - 1) % 4]
        right_vector = HEADING_VECTORS[(self.heading + 1) % 4]

        # PPO also gets global navigation guidance:
        # normalized position, heading, and relative target direction/distance.
        observation = np.array(
            [
                x / (self.width - 1),
                y / (self.height - 1),
                np.cos(heading_angle),
                np.sin(heading_angle),
                target_dx / (self.width - 1),
                target_dy / (self.height - 1),
                self._distance_to_target(self.position) / diagonal,
                self._sensor_encode(self._offset_position(self.position, front_vector, 1)),
                self._sensor_encode(self._offset_position(self.position, left_vector, 1)),
                self._sensor_encode(self._offset_position(self.position, right_vector, 1)),
                self._sensor_encode(self._offset_position(self.position, front_vector, 2)),
            ],
            dtype=np.float32,
        )
        return observation

    def _sensor_encode(self, position: tuple[int, int]) -> float:
        """Convert map contents into simple numeric sensor values."""
        if not self._in_bounds(position):
            return 1.0
        if position == self.target_pos:
            return -0.5

        cell = self._cell_value(position)
        if cell == FREE:
            return 0.0
        if cell == LOW_OBSTACLE:
            return 0.5
        return 1.0

    def _distance_to_target(self, position: tuple[int, int]) -> float:
        """Compute Euclidean distance from the robot to the target."""
        return float(np.hypot(self.target_pos[0] - position[0], self.target_pos[1] - position[1]))

    def _cell_value(self, position: tuple[int, int]) -> int:
        x, y = position
        return int(self.grid[y, x])

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        """Check whether a position is still inside the grid."""
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _offset_position(self, position: tuple[int, int], vector: tuple[int, int], steps: int) -> tuple[int, int]:
        """Move a position along a heading vector by a fixed number of cells."""
        return (position[0] + vector[0] * steps, position[1] + vector[1] * steps)

    def _build_info(self, *, success: bool, collision: bool, blocked: bool, failed_jump: bool) -> dict[str, Any]:
        """Return extra debug information that is useful during inspection."""
        return {
            "action_budget_remaining": self.max_steps - self.step_count,
            "blocked": blocked,
            "collision": collision,
            "distance_to_target": self._distance_to_target(self.position),
            "episode_reward": self.episode_reward,
            "failed_jump": failed_jump,
            "heading": HEADING_NAMES[self.heading],
            "jump_count": self.jump_count,
            "position": self.position,
            "success": success,
        }

    def _build_default_grid(self) -> np.ndarray:
        """Create the first hand-authored warehouse layout used for debugging."""
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # A fixed map is easier to debug than a random one while we are learning the basics.
        wall_positions = {
            (5, 4),
            (5, 6),
            (6, 2),
            (6, 3),
            (6, 7),
            (6, 8),
        }
        low_obstacle_positions = {
            (4, 5),
            (3, 2),
            (7, 6),
        }

        for x, y in wall_positions:
            grid[y, x] = WALL
        for x, y in low_obstacle_positions:
            grid[y, x] = LOW_OBSTACLE

        return grid
