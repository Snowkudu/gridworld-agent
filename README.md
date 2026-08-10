# GridWorld ML/RL Agent

A staged machine-learning project that begins with supervised imitation of a BFS expert policy and later expands into CNN and reinforcement-learning agents.

## Current status

- **P1 complete:** the environment, oracle policy, canonical dataset, dataset validator, and contract tests are frozen as version 1.
- **P2 complete:** the supervised MLP baseline, experiment sweep, checkpoint reconstruction, training diagnostics, and automated tests are implemented.
- **P3 complete:** closed-loop MLP evaluation, rescue-policy diagnostics, headless episode metrics, and a minimal Pygame rollout viewer are implemented.
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

## P3 closed-loop evaluation

P3 evaluates frozen P2 checkpoints as policies inside the environment rather than as independent state classifiers.

The evaluator compares each raw MLP action with the BFS oracle. When they agree, the MLP action is executed. When they disagree, the evaluator rejects the raw action and selects a random legal alternative that is different from the rejected action. The oracle therefore acts as a disagreement gate, but it does not choose the rescue action.This is to ensure the model can eventualy visit neighboring states that it may have not if not for the BFS absolute dictation,leading to the conclusion that the labeled dataset used to train this model performs rather poorly on this configuration .

This rescue rule was kept deliberately simple. Its purpose is to prevent deterministic evaluator-created softlocks without adding planning, memory, goal-directed heuristics, or retraining. A model can still wander until the 50-step rollout limit.

### P3 Closed-Loop MLP Comparison

| P3 rollout metric | **MLP 256×128**<br>`bs16 · lr3e-4 · dropout 0.10` | **MLP 64×64**<br>`bs32 · lr3e-4 · dropout 0.10` |
|---|---:|---:|
| Episodes | 50 | 50 |
| Assisted success rate | **80%** | 76% |
| Fully autonomous success rate | **4%** | 2% |
| Timeout rate | 20% | 24% |
| Oracle agreement | **23.0%** | 19.8% |
| Random rescue rate | 77.0% | **80.2%** |
| Avg. steps on successful episodes | 22.1 | **20.1** |

The `256×128` model is the selected P2 checkpoint. The `64×64` model is an alternate checkpoint used as a reference. Both show the same qualitative closed-loop failure pattern: very low autonomous success and heavy dependence on disagreement-triggered random rescue.

The assisted success rate should not be interpreted as raw MLP performance. The selected model completes only `2/50` episodes without rescue. The higher assisted rate shows that simple stochastic escape can break repeated failure loops, not that the MLP has acquired planning ability.

### P3 findings

- Offline P2 label and prediction distributions are approximately balanced across all four actions.
- Fresh GridWorld states also produce a varied action distribution, ruling out a global class-0 collapse or poor logit selection.
- During self-generated rollouts, the MLP can enter narrow state regions where the same legal action is repeatedly preferred leading to a softlock behaviour.
- Legal moves alone do not prevent recurrent failure loops.
- The MLP has no memory of previous states or actions and no mechanism for planning over future consequences.
- Errors compound because the model's own decisions change the distribution of states it subsequently observes.
- A second MLP checkpoint reproduces the same broad failure pattern, so the behavior is not isolated to the selected checkpoint.

P3 therefore preserves the poor autonomous rollout performance as a result rather than modifying the P2 training process. The next phase tests whether a CNN's spatial inductive bias improves supervised state-to-action learning and closed-loop behavior.

### Pygame viewer

The minimal Pygame viewer is diagnostic only. It renders the grid and a side panel with rollout information so model decisions, oracle comparisons, rescue behavior, success, and timeout states can be inspected visually. Pygame runs do not write evaluation JSON.

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

Run the canonical P3 headless evaluation:

```powershell
python -m environment.evaluate --mode headless --checkpoint artifacts/p2_mlp/mlp_256_128_bs16_lr3e-4_do0.10/best_model.pt --episodes 50 --seed 999
```

Run one P3 Pygame rollout:

```powershell
python -m environment.evaluate --mode pygame --checkpoint artifacts/p2_mlp/mlp_256_128_bs16_lr3e-4_do0.10/best_model.pt --episodes 1 --seed 999
```

Run all tests:

```powershell
python -m pytest -v
```

Run Ruff checks:

```powershell
python -m ruff check .
python -m ruff format --check .
```

## Roadmap

- [x] P1: environment and dataset contract
- [x] P2: supervised MLP baseline and training metrics
- [x] P3: evaluation and minimal Pygame rollout viewer
- [ ] P4: CNN implementation and MLP/CNN comparison
- [ ] P5: DQN reinforcement learning
- [ ] P6: supervised-to-RL transfer experiments
