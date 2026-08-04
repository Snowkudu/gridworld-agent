# GridWorld ML/RL Agent

A staged machine-learning project that begins with supervised imitation of a BFS expert policy and later expands into CNN and reinforcement-learning agents.

## Current status

- **P1 complete:** the environment, oracle policy, canonical dataset, dataset validator, and contract tests are frozen as version 1.
- **P2 complete:** the supervised MLP baseline, experiment sweep, checkpoint reconstruction, training diagnostics, and automated tests are implemented.
- **Current test result:** `45 passed`.

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

## P2 supervised MLP baseline

P2 trains an MLP to imitate the BFS expert policy using the canonical labelled dataset.

### Method

- Deterministic 80/10/10 train, validation, and test split
- Configurable fully connected MLP with ReLU activations
- Cross-entropy classification loss
- Adam optimizer
- Validation-loss checkpoint selection
- Early stopping
- Fixed split and experiment seeds
- Checkpoint reconstruction from the saved model configuration
- Final test evaluation only after model selection

### Selected configuration

| Parameter | Value |
|---|---:|
| Hidden layers | `256, 128` |
| Batch size | `16` |
| Learning rate | `3e-4` |
| Dropout | `0.10` |
| Weight decay | `0.0` |
| Maximum epochs | `50` |
| Early-stopping patience | `8` |
| Minimum improvement | `1e-4` |

### Results

| Metric | Result |
|---|---:|
| Best epoch | `8` |
| Validation loss | `1.1248` |
| Validation accuracy | `47.03%` |
| Test loss | `1.1486` |
| Test accuracy | `45.12%` |
| Majority-class baseline | `26.12%` |

The MLP performs above the majority-class baseline and learns useful state-to-action structure. Classification accuracy alone does not show whether the policy can complete full episodes, so P3 evaluates the selected model through environment rollouts and adds a minimal Pygame viewer for inspecting failures.

The compact committed result is stored in:

```text
results/mlp_final_results.json
```

Generated checkpoints and raw experiment artifacts are excluded from Git.

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

Run the frozen P2 configuration:

```powershell
python -m training.train_mlp
```

Run the full P2 experiment sweep:

```powershell
python -m training.train_mlp --config-set sweep
```

Run all tests:

```powershell
python -m pytest -v
```

## Roadmap

- [x] P1: environment and dataset contract
- [x] P2: supervised MLP baseline and training metrics
- [ ] P3: evaluation and minimal Pygame rollout viewer
- [ ] P4: CNN implementation and MLP/CNN comparison
- [ ] P5: DQN reinforcement learning
- [ ] P6: supervised-to-RL transfer experiments
