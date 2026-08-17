from dataclasses import dataclass
from pathlib import Path
import random

import torch

from agents.dqn import DQNAgent

from configs.dqn_experiments import EXPERIMENTS,EVAL_CONFIG,REWARD_LABELS
from environment.environment import init_world
from environment.rewards import get_reward_fn
import subprocess
import sys
from pathlib import Path
from collections import Counter


@dataclass
class EvaluationResult:
    episodes: int
    successes: int
    success_rate: float
    timeout_rate: float
    mean_return: float
    mean_steps: float
    mean_success_steps: float | None
    action_counts: dict[int, int]
    illegal_rate: float
    two_cycle_rate: float


def export_run_tensorboard(config: dict) -> None:
    tensorboard_dir = (
        Path("artifacts")
        / "p5_dqn"
        / config["name"]
        / "tensorboard"
    )

    subprocess.run(
        [
            sys.executable,
            "utils/export_tensorboard_scalars.py",
            "--input",
            str(tensorboard_dir),
        ],
        check=False,
    )

def evaluate_dqn(
    agent: DQNAgent | None,
    config: dict,
    policy: str = "greedy",
) -> EvaluationResult:
    reward_fn = get_reward_fn(config["reward"])

    successes = 0
    timeouts = 0
    returns = []
    steps = []
    success_steps = []
    action_counts = Counter()
    total_actions = 0
    illegal_actions = 0
    repeated_states = 0
    old_epsilon = None
    two_cycles=0

    if policy == "greedy":
        if agent is None:
            raise ValueError("Greedy policy requires a DQNAgent")

        agent.online.eval()
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0

    else:
        raise ValueError(f"Unknown policy: {policy}")

    try:
        with torch.no_grad():
           for episode_idx, seed in enumerate(config["seeds"]):
                
                env = init_world(
                    seed=seed,
                    max_steps=config["max_steps"],
                    reward_fn=reward_fn,
                )
                state_history = [tuple(env.state)]
                seen_states = {tuple(env.state)}
                done = False
                episode_return = 0.0
                while not done:
                    previous_state = tuple(env.state)

                    action = agent.select_action(env)
                    _, reward, done = env.step(action)

                    episode_return += reward

                    current_state = tuple(env.state)
                    if (
                        len(state_history) >= 2
                        and current_state == state_history[-2]
                        and current_state != previous_state
                    ):
                        two_cycles += 1

                    state_history.append(current_state)
                    action_counts[action] += 1
                    total_actions += 1

                    if current_state == previous_state:
                        illegal_actions += 1

                    if current_state in seen_states:
                        repeated_states += 1
                    else:
                        seen_states.add(current_state)
                success = env.state == env.goal_state
                timeout = (
                    env.currentsteps >= env.maxsteps
                    and not success
                )

                successes += int(success)
                timeouts += int(timeout)

                returns.append(episode_return)
                steps.append(env.currentsteps)

                if success:
                    success_steps.append(env.currentsteps)

    finally:
     assert agent is not None
     assert old_epsilon is not None
     agent.epsilon = old_epsilon
    episodes = len(config["seeds"])

    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        timeout_rate=timeouts / episodes,
        mean_return=sum(returns) / episodes,
        mean_steps=sum(steps) / episodes,
        mean_success_steps=(
            sum(success_steps) / len(success_steps)
            if success_steps
            else None
        ),
        action_counts=dict(action_counts),
        illegal_rate=illegal_actions / total_actions,
        two_cycle_rate=two_cycles / total_actions,
    )


def run_expiriment(config:dict,eval_config:dict)-> dict[str,object]:
        from training.train_dqn import train_dqn
        agent=train_dqn(config)
        checkpoint_path = (
            Path("artifacts")
            / "p5_dqn"
            / config["name"]
            / "checkpoint.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=agent.device,
        )

        agent.online.load_state_dict(
            checkpoint["online_state_dict"]
        )
        result=evaluate_dqn(agent,config=eval_config,policy="greedy")
        return {
        "name": config["name"],
        "reward": REWARD_LABELS[config["training"]["reward"]],
        "train_eps": config["training"]["episodes"],
        "train_steps": config["training"]["max_steps"],
        "gamma": config["dqn"]["gamma"],
        "sync": config["dqn"]["target_sync_interval"],
        "success_rate": result.success_rate,
        "timeout_rate": result.timeout_rate,
        "mean_return": result.mean_return,
        "mean_steps": result.mean_steps,
        "mean_success_steps": result.mean_success_steps,
        "illegal_action_rate": result.illegal_rate,
        "cycle2":result.two_cycle_rate,
        "best_ep": checkpoint["episode"],
        "val_success": checkpoint["validation"]["success_rate"],
    }

def print_results(results: list[dict[str, object]]) -> None:
    if not results:
        print("\nNo experiment results.")
        return

    print("\nDQN EXPERIMENT RESULTS")

    name_w = max(
        len("name"),
        max(len(str(result["name"])) for result in results),
    )

    reward_w = max(
        len("reward"),
        max(len(str(result["reward"])) for result in results),
    )

    header = (
        f"{'name':<{name_w}}  "
        f"{'reward':<{reward_w}}  "
        f"{'train_eps':>9}  "
        f"{'train_cap':>9}  "
        f"{'gamma':>6}  "
        f"{'sync':>6}  "
        f"{'success':>8}  "
        f"{'timeout':>8}  "
        f"{'return':>8}  "
        f"{'steps':>8}  "
        f"{'succ_steps':>10}  "
        f"{'illegal':>8}  "
        f"{'cycle2':>8}  "
        f"{'best_ep':>8}  "
        f"{'val_success':>11}"
    )

    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for result in results:
        success_steps = result["mean_success_steps"]

        success_steps_text = (
            "-"
            if success_steps is None
            else f"{success_steps:.1f}"
        )

        print(
            f"{str(result['name']):<{name_w}}  "
            f"{str(result['reward']):<{reward_w}}  "
            f"{result['train_eps']:>9}  "
            f"{result['train_steps']:>9}  "
            f"{result['gamma']:>6.2f}  "
            f"{result['sync']:>6}  "
            f"{result['success_rate']:>8.1%}  "
            f"{result['timeout_rate']:>8.1%}  "
            f"{result['mean_return']:>8.1f}  "
            f"{result['mean_steps']:>8.1f}  "
            f"{success_steps_text:>10}  "
            f"{result['illegal_action_rate']:>8.1%}  "
            f"{result['cycle2']:>8.1%}  "
            f"{result['best_ep']:>8}  "
            f"{result['val_success']:>11.1%}"
        )

    print("=" * len(header))


def eval_checkpoint(config: dict) -> EvaluationResult:
    checkpoint_path = (
        Path("artifacts")
        / "p5_dqn"
        / config["name"]
        / "checkpoint.pt"
    )

    agent = DQNAgent(
        config_online=config["cnn"],
        config_target=config["cnn"],
        replay_capacity=config["dqn"]["replay_capacity"],
        batch_size=config["dqn"]["batch_size"],
        gamma=config["dqn"]["gamma"],
        learning_rate=config["dqn"]["learning_rate"],
        epsilon_start=config["dqn"]["epsilon_start"],
        epsilon_min=config["dqn"]["epsilon_min"],
        epsilon_decay=config["dqn"]["epsilon_decay"],
        target_sync_interval=config["dqn"]["target_sync_interval"],
        epsilon_update_interval=config["dqn"]["epsilon_update_interval"],
        device="cuda",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
    )
    print(config["name"], config["dqn"]["gamma"], checkpoint_path)
    print(checkpoint["config"]["name"])
    print(checkpoint["config"]["dqn"]["gamma"])
    agent.online.load_state_dict(
        checkpoint["online_state_dict"]
    )

    result = evaluate_dqn(
        agent,
        config=EVAL_CONFIG,
    )

    print(
        config["name"],
        f"best_ep={checkpoint['episode']}",
        f"val={checkpoint['validation']['success_rate']:.1%}",
        result,
    )

    return result

def main():
    help=eval_checkpoint(EXPERIMENTS[1])
    
    results=[]
    for config in EXPERIMENTS:
        try:
            result=run_expiriment(config=config,eval_config=EVAL_CONFIG)
            results.append(result)
        except Exception as exc:
            print(
                f"[FAILED] {config['name']}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            export_run_tensorboard(config)
    print_results(results)
    

if __name__ == "__main__":
    main()