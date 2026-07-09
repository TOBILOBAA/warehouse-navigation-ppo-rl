# Warehouse Navigation MDP Blueprint

## Purpose

This document defines the **first buildable version** of the project.

It is intentionally:

- simple,
- grid-based,
- easy to debug,
- easy to explain.

We are treating this as **Version 0** of the full project.

Later, once this works, we will scale it into a more realistic continuous-space navigation environment.

## 1. Problem Statement

Train a robot agent to navigate from a start location to a target location in a warehouse while avoiding walls and handling low obstacles that may be jumped.

Important framing:

- this is a **navigation problem**,
- the target may be represented by a flag in demos,
- but the target is conceptually just a **goal location**.

## 2. First-Version Assumptions

To keep the first version learnable, we will use these assumptions:

- The environment is a **2D grid world**, not continuous space yet.
- The target position is fixed for the first baseline.
- The robot starts from a fixed entrance cell.
- The robot has 4 headings:
  - north,
  - east,
  - south,
  - west.
- The action space is discrete.
- The map is small enough to render in text during debugging.

These are not permanent limitations. They are training wheels that help us build the system correctly.

## 3. Version 0 Environment Specification

### Map

- Grid size: `10 x 10`
- Cell types:
  - `0 = free`
  - `1 = wall`
  - `2 = low obstacle`
- Start position: `(1, 5)`
- Start heading: `east`
- Target position: `(8, 5)`
- Max steps per episode: `80`

Why this setup:

- it is large enough to require navigation,
- small enough to debug manually,
- and easy to visualize in text.

### What `10 x 10` really means for decision-making

A `10 x 10` grid has:

- `100` total cells,
- up to `4` possible headings at each cell,
- so up to about `400` position-heading states before even considering obstacle context.

But the more practical number for us is:

- **the robot gets at most `80` action decisions in one episode**

That means in one run, the agent can choose up to `80` actions such as:

- move forward,
- turn left,
- turn right,
- jump.

So when we say "can the robot solve the task in time?", for Version 0 we really mean:

- **can it reach the target within `80` decisions?**

That `80` is the episode budget.

It is not saying the robot only has 80 total possibilities in the whole environment.
It is saying:

- in a single attempt,
- the agent gets 80 chances to choose what to do next.

### Example mental model

The agent starts near the left side of the warehouse and must move toward a target near the right side. Some paths are blocked by walls, and some low obstacles can be crossed only with `jump`.

And yes, we will absolutely be able to **look at the environment with our eyes**.

We will start with:

- a text grid renderer for debugging,

and later add:

- a simple graphical renderer.

That visual feedback is important because in RL we do not want to trust numbers blindly. We want to see whether the robot is actually behaving the way we think it is.

### What the 7-action demo tells us

Our hand-written demo route reaches the target in `7` actions.

That does tell us something important:

- **Version 0 is intentionally simple**
- the first map is not meant to be a hard benchmark
- it is meant to be a clean learning and debugging environment

This is a good thing at this stage.

Why:

- if the first environment is too hard, we will not know whether PPO is failing because of:
  - bad observations,
  - bad rewards,
  - bad transitions,
  - or simply because the task is too difficult too early

So the 7-action route does suggest the map is relatively easy.
That is acceptable for Version 0 because the current goal is:

- prove the RL loop works,
- prove the environment logic works,
- and give ourselves a baseline before adding more complexity

Later we can increase complexity by:

- adding more walls,
- adding more jumpable obstacles,
- randomizing the target,
- randomizing the start,
- increasing maze-like structure,
- or reducing how much target information the policy receives

## 4. Agent

The agent is the learning robot.

Its job is to choose an action at every step based only on its observation.

The policy we train later with PPO will map:

`observation -> action`

## 5. Observation Design

### Observation principle

The observation should give the robot enough information to navigate without handing it a full perfect map.

### Observation vector for Version 0

We will use this numeric vector:

1. `x_norm`
2. `y_norm`
3. `cos_heading`
4. `sin_heading`
5. `target_dx_norm`
6. `target_dy_norm`
7. `target_distance_norm`
8. `front_cell_type`
9. `left_cell_type`
10. `right_cell_type`
11. `front_two_cell_type`

### Meaning

- `x_norm`, `y_norm`
  - robot position normalized to `[0, 1]`
- `cos_heading`, `sin_heading`
  - heading encoded numerically without wraparound problems
- `target_dx_norm`, `target_dy_norm`
  - relative target direction in grid coordinates, normalized
- `target_distance_norm`
  - Euclidean distance to target, normalized by grid diagonal
- `front_cell_type`, `left_cell_type`, `right_cell_type`
  - the immediate neighboring cell types relative to the robot
- `front_two_cell_type`
  - the landing cell type two cells ahead, useful for jump decisions

### Cell-type encoding for sensors

For Version 0 we will encode sensed cell types as:

- `0.0 = free`
- `0.5 = low obstacle`
- `1.0 = wall or out of bounds`
- `-0.5 = target cell`

Why this is okay for now:

- it is simple,
- easy to inspect during debugging,
- and enough for a first PPO baseline.

Later we may upgrade this to one-hot or multi-channel local sensing.

## 6. Action Space

We will use a discrete action space:

1. `0 = forward`
2. `1 = turn_left`
3. `2 = turn_right`
4. `3 = jump`

### Action semantics

#### `forward`

- Move one cell in the current heading direction.

#### `turn_left`

- Rotate heading by `-90 degrees`.
- Position does not change.

#### `turn_right`

- Rotate heading by `+90 degrees`.
- Position does not change.

#### `jump`

- If the cell directly ahead is a low obstacle and the landing cell is valid, move two cells forward.
- Otherwise the jump fails and the robot stays in place with a penalty.

### Why turning matters even when position does not change

Turning does not move the robot to a new cell, but it still changes the robot's state.

That matters because heading determines:

- what "forward" means on the next step,
- what the robot sees as front, left, and right,
- whether a move or jump is sensible from the current pose.

So turning is not a wasted action.
It is a way of reorienting the robot for:

- future movement,
- future observation,
- and better decision-making.

## 7. Transition Rules

These define how the world changes after an action.

### Forward rules

- If front cell is free, move into it.
- If front cell is the target, move into it and finish successfully.
- If front cell is a low obstacle, movement is blocked, the robot stays in place, and receives a penalty.
- If front cell is a wall or outside bounds, count as collision, the robot stays in place, and receives a penalty.

### Turn rules

- Turning only changes heading.
- No position change happens.

### Jump rules

- If front cell is a low obstacle and the landing cell is free, jump over it.
- If front cell is a low obstacle and the landing cell is the target, jump and finish successfully.
- If the jump conditions are not satisfied, the robot stays in place and receives a failed-jump penalty.

This gives us a clear distinction between:

- **collision**,
- **blocked movement**,
- **failed jump**.

That distinction helps both learning and debugging.

Important design decision for Version 0:

- **collision does not end the episode**
- **failed jump does not end the episode**

Why:

- the task is navigation under a decision budget,
- the robot should be allowed to make a mistake and recover,
- and we want it to learn rerouting behavior, not only avoid immediate failure.

## 8. Reward Function

### Reward formula

At each step:

`reward = progress_reward + step_penalty + event_reward`

### Components

#### Progress reward

- `progress_reward = old_distance_to_target - new_distance_to_target`

This means:

- moving closer gives positive reward,
- moving farther gives negative reward,
- standing still gives near-zero progress reward.

#### Step penalty

- `step_penalty = -0.01`

This encourages shorter paths.

#### Event rewards

- Reach target: `+10.0`
- Collision with wall/out of bounds: `-0.50`
- Forward into low obstacle: `-0.20`
- Failed jump: `-0.30`

### Why this reward design is good for Version 0

- It strongly rewards success.
- It punishes bad actions without immediately ending recovery.
- It gives dense learning signal through distance progress.
- It discourages wasting time.
- It discourages spamming jump.

### How to read the reward formula

At each step:

`reward = progress_reward + step_penalty + event_reward`

This means the total score for one action is built from three pieces:

1. `progress_reward`
   - Did the robot get closer to the target or farther away?
2. `step_penalty`
   - Did the robot spend one more decision from its limited budget?
3. `event_reward`
   - Did something important happen, like reaching the target, colliding, or failing a jump?

So the reward is not just "good" or "bad."
It is a small accounting system that answers:

- Did you make progress?
- Did you waste time?
- Did a notable event happen?

## 9. Episode Termination

The episode ends if any of these happen:

1. The robot reaches the target.
2. The robot exceeds `max_steps`.

For Version 0:

- collision is **non-terminal**
- blocked forward attempts are **non-terminal**
- failed jumps are **non-terminal**

The robot keeps trying until it either:

- reaches the target,
- or runs out of decision budget.

## 10. Success Metrics

We will track these metrics during evaluation:

1. `success_rate`
2. `average_episode_reward`
3. `average_steps_per_successful_episode`
4. `collision_rate`
5. `average_jumps_per_episode`

These metrics matter more than reward alone.

### What these metrics actually tell us

#### `success_rate`

Out of all episodes, how many ended with the robot reaching the target?

This answers:

- is the robot actually solving the navigation task?

#### `average_episode_reward`

The average total reward collected per episode.

This answers:

- is overall behavior improving according to the reward design?

It is useful, but we do not trust it by itself because reward can sometimes improve for the wrong reasons.

#### `average_steps_per_successful_episode`

Among successful episodes only, how many decisions did the robot take on average before reaching the target?

This answers:

- when the robot succeeds, is it efficient or wasteful?

#### `collision_rate`

How often the robot attempts actions that cause collision events.

This answers:

- is the robot navigating safely,
- or is it still bumping into bad choices too often?

#### `average_jumps_per_episode`

How many jumps the robot uses on average in each episode.

This answers:

- is jump being used meaningfully,
- or is the policy overusing or underusing it?

## 11. Why Grid First

We are starting with grid world because it makes the core RL questions easier to learn:

- Is the observation sufficient?
- Are the actions well defined?
- Does reward shaping work?
- Does PPO improve behavior?
- Are collisions and goal conditions implemented correctly?

### What those diagnostic questions really mean

#### Is the observation sufficient?

Does the robot receive enough information to make good navigation choices?

This question checks whether failure is happening because:

- the robot is under-informed,
- not because the algorithm is bad.

#### Are the actions well defined?

Does each action have a clear, consistent effect in the environment?

This question checks whether the agent is learning in a stable world with understandable controls.

#### Does reward shaping work?

Does the reward system actually encourage the behavior we care about?

This question checks whether the robot is being taught the right lesson.

#### Does PPO improve behavior?

As training goes on, does the policy become more successful, efficient, and reliable?

This question checks whether learning is genuinely happening.

#### Are collisions and goal conditions implemented correctly?

Do success, failure, and penalty events happen exactly when we intend them to?

This question checks whether the environment rules match the task we think we designed.

If we cannot solve the grid version, the continuous version will only become more confusing.

## 12. How We Will Scale To Continuous Space Later

Once Version 0 works, the next upgrade path is:

1. Replace grid cell positions with continuous `(x, y)` coordinates.
2. Replace 4-direction headings with continuous angle `theta`.
3. Keep the same navigation objective.
4. Revisit whether actions stay discrete or become continuous controls.
5. Upgrade obstacle sensing from local cell checks to range-style sensing.

This is the right progression because the problem definition stays the same while the physics and representation become more realistic.

## 13. Implementation Targets

These are the first code artifacts this blueprint implies:

- `env/warehouse_navigation_env.py`
- `configs/ppo_config.yaml`
- `scripts/train_ppo.py`
- `scripts/eval_ppo.py`
- `scripts/render_demo.py`

## 14. What We Are Deciding Today

For the first implementation, we are locking in:

- grid world,
- discrete actions,
- fixed start,
- fixed target,
- shaped rewards,
- PPO baseline,
- text rendering before graphics.

That gives us a clean and teachable first milestone.
