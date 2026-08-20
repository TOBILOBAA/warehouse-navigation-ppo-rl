# Deep RL Warehouse Navigation Workflow

## 1. What This Project Really Is

This project is a **robot navigation RL project** wrapped in a simple warehouse story:

- A robot starts near the warehouse entrance.
- A target location sits near the far end.
- The robot must learn how to reach that target.
- The world contains:
  - empty space,
  - walls that must be avoided,
  - low obstacles that can be jumped.

The important modeling idea is this:

- the **task is navigation**,
- the **flag is only one example of a target marker**,
- later the target could be a pallet bay, pickup zone, charging dock, loading station, or any goal position.

Under the hood, the project is made of two parts:

1. **The environment**
   - We define the warehouse, the robot, the obstacles, the sensors, the actions, the rewards, and the episode ending rules.
2. **The learning algorithm**
   - We use PPO to learn a policy that maps observations to actions.

If you understand those two parts deeply, you understand most beginner-to-intermediate deep RL robotics projects.

## 2. What The Design Document Specifies

The design document defines:

- A custom `Gymnasium` environment that we will rename to `WarehouseNavigationEnv`.
- A discrete action space:
  - `forward`
  - `turn_left`
  - `turn_right`
  - `jump`
- A state vector containing:
  - robot position,
  - orientation,
  - target distance,
  - target bearing,
  - local obstacle readings.
- A shaped reward:
  - big positive for reaching the target,
  - big negative for collision,
  - small positive when moving closer,
  - small negative when moving away,
  - small time penalty per step.
- A PPO training pipeline using Stable Baselines3.
- Evaluation and demo scripts.
- Documentation and portfolio packaging.

## 3. The Real Learning Goal

This is not only about "making PPO work."

The real skill we are building is:

**learning how to translate a robotics problem into an RL problem**

That means being able to answer:

- What is the observation?
- What is the action space?
- What is the reward?
- What ends an episode?
- What should be simple first, and what should be realistic later?
- How do we tell whether learning is actually happening?

## 4. Our A-to-Z Workflow

We will use this exact order.

### Step 1. Translate the idea into an RL formulation

Goal:
Turn the story into a precise Markov Decision Process.

We will define:

- agent,
- environment,
- observation,
- action space,
- reward,
- terminal conditions,
- success metrics.

Why this matters:
Most RL problems fail because the formulation is vague, not because PPO is bad.

Deliverable:
A one-page MDP blueprint we both understand.

### Step 2. Build the grid-world version first

Goal:
Create a minimal warehouse that is easy to debug and easy to learn from.

Version 0 should be:

- small 2D grid,
- fixed start position,
- fixed target position,
- a few walls,
- maybe one jumpable obstacle,
- text rendering first.

Why this matters:
If we start too complex, we will not know whether failures come from the environment, reward, or training.

Deliverable:
A tiny environment where we can manually step the robot around.

### Step 3. Implement the Gymnasium interface correctly

Goal:
Make the environment structurally valid.

We will implement:

- `__init__`
- `reset`
- `step`
- `render`
- `close`

Why this matters:
RL libraries are very strict about environment contracts. If this layer is shaky, training will become confusing fast.

Deliverable:
A working `WarehouseNavigationEnv` that returns valid observations, rewards, done flags, and info dictionaries.

Current status:

- `reset()` and `step()` follow Gymnasium's expected signatures
- `action_space` and `observation_space` are defined
- the real dependency stack is installed in the project venv
- Gymnasium's environment checker passes

This means Step 3 is complete enough for us to confidently move on to environment debugging and sanity testing.

### Step 4. Implement movement and world rules

Goal:
Make actions change the world consistently.

We will define:

- how forward movement works,
- how turning changes heading,
- how jumping works,
- what counts as collision,
- what happens at map boundaries,
- which mistakes are recoverable versus episode-ending.

Why this matters:
The agent can only learn from world dynamics that are consistent and predictable.

Deliverable:
Reliable transition logic.

### Step 5. Design the observation space carefully

Goal:
Give the robot enough information to learn, but not magical information.

We will decide:

- exact observation vector layout,
- whether coordinates are absolute or normalized,
- how target direction is encoded,
- how local obstacle sensing is encoded.

Why this matters:
Observation design is one of the biggest levers in RL. Too little information makes learning impossible. Too much unrealistic information makes the task less meaningful.

Deliverable:
A stable observation vector and `observation_space`.

### Step 6. Design and tune the reward function

Goal:
Teach the agent what "good navigation" means.

We will implement:

- goal reward,
- collision penalty,
- progress reward,
- time penalty,
- optional failed-jump penalty.

Why this matters:
Reward shaping controls behavior. Even small changes can make the agent reckless, lazy, or stuck in local tricks.

Deliverable:
A reward function that encourages safe, efficient progress toward the target.

For Version 0, collision and failed jump are penalized but non-terminal so the robot can recover and keep navigating within its step budget.

### Step 7. Debug the environment before training

Goal:
Prove the environment makes sense before PPO touches it.

We will test:

- random actions,
- hand-crafted action sequences,
- edge cases,
- reward sanity,
- terminal conditions.

Why this matters:
If we skip this, we may waste hours blaming PPO for environment bugs.

Deliverable:
An environment we trust.

Current status:

- a repeatable sanity suite exists in `tests/test_env_sanity.py`
- wall collision attempts are checked
- blocked forward by low obstacle is checked
- failed and successful jump behavior is checked
- turning/observation change is checked
- max-step truncation is checked
- target-reaching termination is checked

This means the environment is no longer only "implemented"; it is now sanity-tested against the core MDP behaviors we care about before PPO training.

### Step 8. Train a PPO baseline

Goal:
Get the first learnable policy working.

We will set up:

- dependencies,
- `train_ppo.py`,
- PPO hyperparameters,
- logging,
- model saving.

Why this matters:
This is the first full RL loop: observation -> action -> reward -> policy update.

Deliverable:
A baseline PPO model that shows measurable improvement over time.

Current status:

- `baseline_v3` became our reliable simple-grid baseline
- `baseline_v4` proved that stronger deterministic behavior survived a harder fixed map
- `baseline_v5a_map1` and `baseline_v5a_map2` both reached perfect deterministic and stochastic evaluation on richer fixed layouts
- those `v5a` runs also taught us an important lesson:
  the policy stayed strong, but it collapsed onto one very confident route instead of showing broad route diversity

This means the next phase is no longer "can PPO solve a richer fixed map?"
It is "can the policy still navigate well when the episode layout changes enough to punish memorization?"

### Step 9. Evaluate behavior, not just reward

Goal:
Check whether the robot is actually becoming a better navigator.

We will measure:

- success rate,
- average steps,
- collision rate,
- episode reward,
- jump usage.

Why this matters:
A rising reward curve can still hide bad behavior. We want real navigation improvement.

Deliverable:
An evaluation script and a short metrics summary.

Current status:

- evaluation already includes deterministic and stochastic rollouts
- visual episode reports are part of the inspection workflow
- `v5a` showed stable success, but also revealed that stable success is not the same thing as flexible navigation

So the next evaluation questions are:

- does randomized start position break the learned behavior?
- does randomized target position break the learned behavior?
- if we redesign the map so the jump shortcut is not obviously best, does the policy still behave intelligently?

We will track those questions as:

- `v5b`: randomize starts while keeping the target fixed
- `v5c`: randomize targets while keeping the start fixed
- `v5d`: keep fixed endpoints, but redesign the map so jump is useful without being automatically dominant
- `v5e`: add an anti-stagnation reward penalty so the policy is discouraged from spinning in place when route choice gets hard
- `v5f`: combine randomized targets with the anti-stagnation penalty to test whether the fix transfers to a harder generalization setting

Latest status:

- `v5b` stayed strong in both deterministic and stochastic evaluation
- `v5c` exposed a major deterministic weakness:
  the policy could often solve the task stochastically, but deterministic rollouts frequently timed out
- `v5d` exposed the failure mode even more clearly:
  deterministic behavior collapsed into repeated left-right turning loops at one cell
- `v5e` targeted that exact issue with a focused anti-stagnation penalty
- `baseline_v5e_anti_loop` recovered clean deterministic behavior on the hard balanced-routes map:
  deterministic success rate `1.0`, timeout rate `0.0`, stuck rate `0.0`
- `v5f` confirmed that the same focused fix transfers back to randomized targets:
  deterministic success rate `1.0`, stochastic success rate `1.0`, timeout rate `0.0`, stuck rate `0.0`

### Step 10. Scale from grid to continuous navigation and document the system

Goal:
Turn the simple learnable prototype into a stronger robotics-style navigation project.

We will create:

- a continuous-space variant after the grid baseline works,
- a better renderer,
- plots,
- demo rollouts,
- README,
- project notes,
- future extension ideas.

Why this matters:
A strong robotics AI engineer does not only train models. They know how to start simple, prove the idea, and then scale realism in a controlled way.

Deliverable:
A portfolio-ready navigation project with room for future ROS2/Gazebo expansion.

## 5. The Most Important Nuances To Understand

These are the ideas I want you to keep in your head while we build.

### Environment design comes before algorithm obsession

If the environment is poorly designed, no amount of PPO tuning will save the project.

### Start simple, then add realism

We should not begin with a "realistic warehouse." We should begin with a learnable warehouse.

That means:

- first a grid world,
- then a richer grid world,
- then a continuous-space version.

### Reward shaping is a teaching signal

The reward is not just scoring. It is how we communicate behavior preferences to the agent.

### Evaluation must match the real task

If the task is navigation, then success rate and collision rate matter, not only total reward.

### Debugging RL is mostly systems thinking

When learning fails, the bug is often in one of these:

- observations,
- rewards,
- termination logic,
- action semantics,
- normalization,
- hyperparameters.

## 6. Our Immediate Build Order

This is the exact practical order I recommend for implementation.

1. Create the repo structure.
2. Write the minimal environment shell.
3. Add fixed map, start, and target.
4. Add movement and turning.
5. Add jump behavior.
6. Add observations.
7. Add reward and done logic.
8. Add text rendering.
9. Run manual and random-agent tests.
10. Add PPO training.
11. Add evaluation metrics.
12. Add graphical demo and documentation.
13. Upgrade to continuous-space navigation.

## 7. What We Should Do Next

The next best step is:

**Step 1: write the exact MDP blueprint for this project**

That means we will make the following explicit:

- exact map representation,
- exact observation vector,
- exact action definitions,
- exact reward formula,
- exact episode ending rules,
- exact success metrics for the first training run.

That blueprint now lives in `docs/warehouse_navigation_mdp.md`.

Once that is written clearly, the implementation becomes much more straightforward.

## 8. Naming Conventions We Will Use

To keep the whole project consistent, we will use these names going forward:

- Project name: `Deep RL Warehouse Navigation`
- Environment name: `WarehouseNavigationEnv`
- Repository name: `warehouse-navigation-ppo-rl`
- Goal terminology: `target` or `goal location`
- Example target object: `flag`, `dock`, `pickup zone`, or `station`

This keeps the project framed around navigation, not around one specific target object.
