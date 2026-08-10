from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment.environment import init_world
from environment.viewer import PygameViewer
from models.checkpoint import load_model_from_checkpoint
from policies.oracle import OraclePolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["headless", "pygame"],
        required=True,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=999)
    return parser.parse_args()


def predict_logits(
    model: nn.Module,
    state: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    if state.ndim == 1:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return logits.squeeze(0).cpu()


def action_from_logits(logits: torch.Tensor) -> int:
    return int(logits.argmax().item())


# get a random valid logit
def random_logit(logits: torch.Tensor, env):
    chosen = int(logits.argmax().item())
    legal_actions = [
        action
        for action in range(logits.numel())
        if env.is_legal(action) and action != chosen
    ]
    if legal_actions == []:
        action = chosen
    action = random.choice(legal_actions)
    return action


def run_headless_eval(env, model, device):
    done = False
    steps = random_rescues = oracle_matches = 0
    while not done and steps < env.maxsteps:
        oracle_action = OraclePolicy.select_action(env)
        model_action = action_from_logits(
            predict_logits(model, env.state_vector(), device)
        )
        if model_action == oracle_action:
            oracle_matches += 1
        else:
            model_action = random_logit(
                predict_logits(model, env.state_vector(), device), env
            )
            random_rescues += 1
        _, _, done = env.step(model_action)
        steps += 1

    assert steps == oracle_matches + random_rescues
    success = env.state == env.goal_state
    timeout = steps >= env.maxsteps and not success
    episode_stats = {
        "steps": steps,
        "success": success,
        "timeout": timeout,
        "oracle_matches": oracle_matches,
        "oracle_agreement": oracle_matches / steps if steps else 0.0,
        "random_rescues": random_rescues,
        "rescue_rate": random_rescues / steps if steps else 0.0,
        "fully_autonomous": success and random_rescues == 0,
    }
    return episode_stats


def run_pygame_eval(env, model, device, fps):
    viewer = PygameViewer(env.grid_size)
    done = False
    steps = random_rescues = oracle_matches = 0
    status = "RUNNING"
    viewer.render_grid(env)
    viewer.tick(fps)
    while not done and steps < env.maxsteps:
        if not viewer.process_events():
            break
        model_action = action_from_logits(
            predict_logits(model, env.state_vector(), device)
        )
        oracle_action = OraclePolicy.select_action(env)
        action = model_action
        if model_action == oracle_action:
            oracle_matches += 1
        else:
            action = random_logit(
                predict_logits(model, env.state_vector(), device), env
            )
            random_rescues += 1
            assert action != model_action
            assert env.is_legal(action)

        _, _, done = env.step(action)
        steps += 1
        viewer.render(
            env,
            steps,
            model_action,
            oracle_action,
            action,
            random_rescues,
            status=status,
        )
        viewer.tick(fps)
    status = "SUCCESS" if env.state == env.goal_state else "TIMEOUT"
    viewer.render(
        env,
        steps,
        model_action,
        oracle_action,
        action,
        random_rescues,
        status=status,
    )
    waiting = True
    while waiting:
        waiting = viewer.process_events()
        viewer.tick(30)

    viewer.close()


def main() -> int:
    config = parse_args()
    env = init_world(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model_from_checkpoint(config.checkpoint, device)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    all_episode_stats = []
    if config.mode == "headless":
        for episode in range(config.episodes):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            env = init_world(episode_seed)
            episode_stat = run_headless_eval(env, model, device)
            all_episode_stats.append(episode_stat)

        results = {
            "checkpoint": str(config.checkpoint),
            "episodes": all_episode_stats,
        }
        checkpoint_name = Path(config.checkpoint).parent.name
        output_path = Path("results/p3_eval") / f"{checkpoint_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation results to {output_path}")
    else:
        print("Pygame config")
        run_pygame_eval(env, model, device, 1)

    return 0


if __name__ == "__main__":
    main()
