from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.representation import to_cnn_1ch, to_cnn_3ch
from environment.environment import init_world
from environment.viewer import PygameViewer
from models.checkpoint import load_model_from_checkpoint
from policies.oracle import OraclePolicy




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["headless", "pygame","autonomous"],
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
    config: dict,
    
) -> torch.Tensor:
    model.eval()
    tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if config["model_type"] == "cnn":
        input_ch=int( config["input_ch"])
        if input_ch == 1: tensor=to_cnn_1ch(tensor)
        elif input_ch==3: tensor= to_cnn_3ch(tensor)
        else: raise ValueError("not supported input")
    tensor=tensor.to(device) 
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


def run_headless_eval(env, model, device,config,mode):
    done = False
    disagreements= steps = random_rescues = oracle_matches = 0
    visited_counts: dict[tuple, int] = {}
    repeated_states = 0

    while not done and steps < env.maxsteps:
        state_key = tuple(env.state_vector())
        visited_counts[state_key] = visited_counts.get(state_key, 0) + 1
        if visited_counts[state_key] == 3:
            repeated_states += 1
        oracle_action = OraclePolicy.select_action(env)
        model_action = action_from_logits(
            predict_logits(model, env.state_vector(), device,config)
        )
        if model_action == oracle_action:
            oracle_matches += 1
        else :
            if mode =="autonomous":
                disagreements+=1
            else:
                model_action = random_logit(
                    predict_logits(model, env.state_vector(), device,config), env
                )
                random_rescues += 1
        _, _, done = env.step(model_action)
        steps += 1
    max_state_visits = max(visited_counts.values(), default=0)
    assert steps == oracle_matches + random_rescues+disagreements
    success = env.state == env.goal_state
    timeout = steps >= env.maxsteps and not success
    episode_stats = {
        "mode" : mode,
        "steps": steps,
        "success": success,
        "timeout": timeout,
        "oracle_matches": oracle_matches,
        "disagreements": disagreements,
        "oracle_agreement": oracle_matches / steps if steps else 0.0,
        "random_rescues": random_rescues,
        "rescue_rate": random_rescues / steps if steps else 0.0,
        "repeated_states": repeated_states,
        "max_state_visits": max_state_visits,
        "fully_autonomous": success and random_rescues == 0,
    }
    return episode_stats


def run_pygame_eval(env, model, device, fps,config,mode):
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
            predict_logits(model, env.state_vector(), device,config)
        )
        oracle_action = OraclePolicy.select_action(env)
        action = model_action
        if model_action == oracle_action:
            oracle_matches += 1
        else:
            action = random_logit(
                predict_logits(model, env.state_vector(), device,config), env
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
    env = init_world(config.seed,200)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,checkpoint = load_model_from_checkpoint(config.checkpoint, device)
    model_config=checkpoint["config"]
   
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    all_episode_stats = []

    if config.mode == "headless" or config.mode == "autonomous":
        for episode in range(config.episodes):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            env = init_world(episode_seed)
            episode_stat = run_headless_eval(env, model, device,model_config,config.mode)
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
        run_pygame_eval(env, model, device, 1,model_config)

    return 0


if __name__ == "__main__":
    main()
