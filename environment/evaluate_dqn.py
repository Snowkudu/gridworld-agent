import argparse
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from agents.dqn import DQNAgent
from configs.dqn_experiments import EVAL_CONFIG, P5_STORY_CONFIGS, REWARD_LABELS
from configs.world_tiers import WORLD_TIERS, save_world_tier_results
from environment.environment import init_world
from environment.rewards import get_reward_fn
from training.exploration import make_inertia_transform


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


def load_dqn_checkpoint(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]
    dqn_config = config["dqn"]

    agent = DQNAgent(
        config_online=config["cnn"],
        config_target=config["cnn"],
        replay_capacity=dqn_config["replay_capacity"],
        batch_size=dqn_config["batch_size"],
        gamma=dqn_config["gamma"],
        learning_rate=dqn_config["learning_rate"],
        weight_decay=dqn_config.get("weight_decay", 0.0),
        epsilon_start=dqn_config["epsilon_start"],
        epsilon_min=dqn_config["epsilon_min"],
        epsilon_decay=dqn_config["epsilon_decay"],
        epsilon_update_interval=dqn_config["epsilon_update_interval"],
        target_sync_interval=dqn_config["target_sync_interval"],
        device=device,
    )

    agent.online.load_state_dict(checkpoint["online_state_dict"])

    agent.sync_target()

    return agent, config


def export_run_tensorboard(config: dict) -> None:
    tensorboard_dir = Path("artifacts") / "p5_dqn" / config["name"] / "tensorboard"

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
    q_transform_fn=None,
    inertia_strength: float = 0.0,
) -> EvaluationResult:
    reward_fn = get_reward_fn(
        config["reward"],
        gamma=agent.gamma if agent is not None else config.get("gamma"),
    )
    successes = 0
    timeouts = 0

    returns = []
    steps = []
    success_steps = []

    action_counts = Counter()

    total_actions = 0
    illegal_actions = 0
    repeated_states = 0
    two_cycles = 0

    if policy != "greedy":
        raise ValueError(f"Unknown policy: {policy}")

    if agent is None:
        raise ValueError("Greedy policy requires a DQNAgent")

    # Preserve agent runtime state.
    old_epsilon = agent.epsilon
    old_q_transform = agent.q_transform
    was_training = agent.online.training

    agent.online.eval()
    agent.epsilon = 0.0

    try:
        with torch.no_grad():
            for seed in config["seeds"]:
                env = init_world(
                    seed=seed,
                    max_steps=config["max_steps"],
                    reward_fn=reward_fn,
                    min_solution_steps=config.get("min_solution_steps", 0),
                )

                done = False
                episode_return = 0.0

                previous_action: int | None = None

                initial_state = tuple(env.state)
                state_history = [initial_state]
                seen_states = {initial_state}

                while not done:
                    previous_state = tuple(env.state)

                    # -----------------------------------------
                    # Runtime policy transform
                    # -----------------------------------------
                    if q_transform_fn is not None:
                        agent.q_transform = q_transform_fn(
                            previous_action=previous_action,
                            strength=inertia_strength,
                        )
                    else:
                        agent.q_transform = None

                    # Greedy because epsilon == 0.
                    action = agent.select_action(env)

                    _, reward, done = env.step(action)

                    current_state = tuple(env.state)

                    # -----------------------------------------
                    # Temporal state for inertia
                    # -----------------------------------------
                    if current_state != previous_state:
                        previous_action = action

                    # -----------------------------------------
                    # Episode metrics
                    # -----------------------------------------
                    episode_return += reward

                    action_counts[action] += 1
                    total_actions += 1

                    # Illegal action == no displacement.
                    if current_state == previous_state:
                        illegal_actions += 1

                    # Any state revisit.
                    if current_state in seen_states:
                        repeated_states += 1
                    else:
                        seen_states.add(current_state)

                    # A -> B -> A
                    if (
                        len(state_history) >= 2
                        and current_state == state_history[-2]
                        and current_state != previous_state
                    ):
                        two_cycles += 1

                    state_history.append(current_state)

                # ---------------------------------------------
                # Episode outcome
                # ---------------------------------------------
                success = env.state == env.goal_state
                timeout = env.currentsteps >= env.maxsteps and not success

                successes += int(success)
                timeouts += int(timeout)

                returns.append(episode_return)
                steps.append(env.currentsteps)

                if success:
                    success_steps.append(env.currentsteps)

    finally:
        # Validation should leave the training agent exactly
        # as it found it.
        agent.epsilon = old_epsilon
        agent.q_transform = old_q_transform

        if was_training:
            agent.online.train()

    episodes = len(config["seeds"])

    return EvaluationResult(
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        timeout_rate=timeouts / episodes,
        mean_return=sum(returns) / episodes,
        mean_steps=sum(steps) / episodes,
        mean_success_steps=(
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        action_counts=dict(action_counts),
        illegal_rate=(illegal_actions / total_actions if total_actions else 0.0),
        two_cycle_rate=(two_cycles / total_actions if total_actions else 0.0),
    )


def run_experiment(
    config: dict,
    eval_config: dict,
) -> dict[str, object]:
    from training.exploration import make_inertia_transform
    from training.train_dqn import train_dqn

    agent = train_dqn(config)

    checkpoint_path = Path("artifacts") / "p5_dqn" / config["name"] / "checkpoint.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )

    agent.online.load_state_dict(checkpoint["online_state_dict"])

    # ---------------------------------------------------------
    # Reconstruct the policy state that belonged to this CP
    # ---------------------------------------------------------
    checkpoint_episode = checkpoint["episode"]

    inertia_config = config["training"].get(
        "inertia",
        {},
    )

    inertia_enabled = inertia_config.get(
        "enabled",
        False,
    )
    inertia_start_episode = inertia_config.get(
        "start_episode",
        0,
    )
    inertia_strength = inertia_config.get(
        "strength",
        0.0,
    )

    inertia_active = inertia_enabled and checkpoint_episode >= inertia_start_episode

    q_transform_fn = make_inertia_transform if inertia_active else None

    # ---------------------------------------------------------
    # Held-out evaluation
    # ---------------------------------------------------------
    result = evaluate_dqn(
        agent,
        config=eval_config,
        policy="greedy",
        q_transform_fn=q_transform_fn,
        inertia_strength=inertia_strength,
    )

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
        "cycle2": result.two_cycle_rate,
        "best_ep": checkpoint_episode,
        "val_success": checkpoint["validation"]["success_rate"],
        "inertia": inertia_active,
        "inertia_strength": (inertia_strength if inertia_active else 0.0),
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
        f"{'inertia':>8}"
        f"{'inertia_strength':>8}"
    )

    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for result in results:
        success_steps = result["mean_success_steps"]

        success_steps_text = "-" if success_steps is None else f"{success_steps:.1f}"

        print(
            f"{result['name']!s:<{name_w}}  "
            f"{result['reward']!s:<{reward_w}}  "
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
            f"{result['inertia']:>8%}"
            f"{result['inertia_strength']:>8%}"
        )

    print("=" * len(header))


def eval_checkpoint(config: dict) -> EvaluationResult:
    checkpoint_path = Path("artifacts") / "p5_dqn" / config["name"] / "checkpoint.pt"

    dqn_config = config["dqn"]
    training_config = config["training"]
    inertia_config = training_config.get("inertia", {})

    agent = DQNAgent(
        config_online=config["cnn"],
        config_target=config["cnn"],
        replay_capacity=dqn_config["replay_capacity"],
        batch_size=dqn_config["batch_size"],
        gamma=dqn_config["gamma"],
        learning_rate=dqn_config["learning_rate"],
        epsilon_start=dqn_config["epsilon_start"],
        epsilon_min=dqn_config["epsilon_min"],
        epsilon_decay=dqn_config["epsilon_decay"],
        epsilon_update_interval=dqn_config["epsilon_update_interval"],
        target_sync_interval=dqn_config["target_sync_interval"],
        device="cuda",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )

    agent.online.load_state_dict(checkpoint["online_state_dict"])

    inertia_enabled = inertia_config.get("enabled", False)
    inertia_strength = float(inertia_config.get("strength", 0.0))

    q_transform_fn = make_inertia_transform if inertia_enabled else None

    print(
        config["name"],
        f"gamma={dqn_config['gamma']}",
        f"inertia={inertia_enabled}",
        f"inertia_strength={inertia_strength}",
        checkpoint_path,
    )

    result = evaluate_dqn(
        agent,
        config=EVAL_CONFIG,
        q_transform_fn=q_transform_fn,
        inertia_strength=inertia_strength,
    )

    print(
        config["name"],
        f"best_ep={checkpoint['episode']}",
        f"val={checkpoint['validation']['success_rate']:.1%}",
        result,
    )

    return result


def print_eval_results(results):
    print("WORLD TIER RESULTS")
    print("=" * 105)

    print(
        f"{'tier':<8}"
        f"{'success':>10}"
        f"{'timeout':>10}"
        f"{'return':>10}"
        f"{'steps':>10}"
        f"{'succ_steps':>12}"
        f"{'illegal':>10}"
        f"{'cycle2':>10}"
    )

    print("-" * 105)

    for result in results:
        print(
            f"{result['name']:<8}"
            f"{result['success_rate'] * 100:>9.1f}%"
            f"{result['timeout_rate'] * 100:>9.1f}%"
            f"{result['mean_return']:>10.1f}"
            f"{result['mean_steps']:>10.1f}"
            f"{result['mean_success_steps']:>12.1f}"
            f"{result['illegal_rate'] * 100:>9.1f}%"
            f"{result['two_cycle_rate'] * 100:>9.1f}%"
        )

    print("=" * 105)


def run_world_tier_evaluation(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, config = load_dqn_checkpoint(checkpoint_path, device)

    results = []

    for tier_name, eval_config in WORLD_TIERS.items():
        result = evaluate_dqn(
            agent,
            config=eval_config,
            q_transform_fn=make_inertia_transform,
            inertia_strength=config["training"]["inertia"]["strength"],
        )

        result_dict = asdict(result)
        result_dict["name"] = tier_name
        results.append(result_dict)

    path = Path("artifacts/p5_dqn/results")
    checkpoint_path = Path(checkpoint_path)
    run_name = checkpoint_path.parent.name

    print("\n" + "=" * 80)
    print(f"CHECKPOINT RUN: {run_name}")
    print("=" * 80)

    print_eval_results(results)

    save_world_tier_results(
        results,
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "train",
        help="Train and evaluate configured DQN experiments.",
    )

    tiers = subparsers.add_parser(
        "tiers",
        help="Evaluate a trained checkpoint across world tiers.",
    )

    tiers.add_argument(
        "--checkpoint",
        required=True,
        help="Path to DQN checkpoint.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "train":
        results = []

        for config in P5_STORY_CONFIGS:
            try:
                result = run_experiment(
                    config=config,
                    eval_config=EVAL_CONFIG,
                )
                results.append(result)

            except (FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"[FAILED] {config['name']}: {type(exc).__name__}: {exc}")

            try:
                export_run_tensorboard(config)
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"[EXPORT FAILED] {config['name']}: {type(exc).__name__}: {exc}")

        print_results(results)

    elif args.command == "tiers":
        run_world_tier_evaluation(
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
