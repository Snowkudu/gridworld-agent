# GridWorld ML/RL Agent

A staged machine-learning project that begins with supervised imitation of a BFS expert policy and later expands into CNN and reinforcement-learning agents.

## Current status

P1 is complete: the environment, oracle policy, dataset generator, dataset validator, and initial automated tests are frozen as version 1.

## P1 environment contract

- Grid size: 10×10
- Obstacles: 30 cells
- State values:
  - `-1`: obstacle
  - `0`: empty
  - `1`: agent
  - `2`: goal
- Actions:
  - `0`: up
  - `1`: down
  - `2`: left
  - `3`: right
- Illegal and obstacle moves leave the agent in place.
- Episodes end when the goal is reached or `max_steps` is exhausted.
- Generated maps must be solvable according to BFS.

## Reward contract

| Event | Reward |
|---|---:|
| Goal | `+10.0` |
| Timeout | `-5.0` |
| Boundary or obstacle hit | `-2.0` |
| Legal move closer to goal | `-0.5` |
| Legal move farther or equal | `-1.5` |

Goal completion takes precedence if the goal is reached on the final permitted step.

## Canonical P1 dataset

- Dataset version: `gridworld_dataset_v1`
- Environment version: `gridworld_env_v1`
- Reward version: `manhattan_shaped_v1`
- Episodes: `2000`
- Maximum steps: `200`
- Seed: `123`
- Samples: `16679`
- Solved episodes: `2000`

Array-content SHA-256:

```text
e5e210983c9e3116b6acd6cde81b12f73aff2142a415fdba89dfb32c7e43c75f
```

Generated datasets are excluded from Git and can be reproduced from source.

## Commands

Generate the canonical dataset:

```powershell
python -m scripts.generate_dataset --episodes 2000 --max_steps 200 --seed 123
```

Replace an existing dataset intentionally:

```powershell
python -m scripts.generate_dataset --episodes 2000 --max_steps 200 --seed 123 --overwrite
```

Validate the dataset:

```powershell
python -m scripts.verify data/raw/gridworld_2000ep_200ms_123seed.npz
```

Run tests:

```powershell
python -m pytest -v
```

Current result: `28 passed`.

## TODO:

- P1: environment and dataset contract
- P2: supervised MLP baseline and training metrics
- P3: evaluation and minimal Pygame rollout viewer
- P4: CNN implementation and MLP/CNN comparison
- P5: DQN reinforcement learning
- P6: supervised-to-RL transfer experiments
