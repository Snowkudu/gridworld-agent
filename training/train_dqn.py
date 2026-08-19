from __future__ import annotations

import random

import torch

from agents.dqn import DQNAgent
from environment.environment import init_world
from environment.evaluate_dqn import EvaluationResult, evaluate_dqn
from environment.rewards import get_reward_fn
from training.dqn_transfer import transfer_cnn_to_dqn
from training.exploration import boltzmann_action, make_inertia_transform, resmax_action
from training.novelty import novelty_action
from training.train_dqn_metrics import DQNMetrics
from utils.dqn_logging import DQNLogger
from utils.reproducibility import seed_everything

VALIDATION_CONFIG = {
    "seeds": list(range(20_000, 20_020)),
    "max_steps": 100,
    "reward": "potential_manhattan_position_terminal",
    "validation_interval": 100,
    "min_solution_steps": 5,
}


def is_better(
    candidate: EvaluationResult,
    best: EvaluationResult | None,
) -> bool:
    if best is None:
        return True

    if candidate.success_rate != best.success_rate:
        return candidate.success_rate > best.success_rate

    if candidate.two_cycle_rate != best.two_cycle_rate:
        return candidate.two_cycle_rate < best.two_cycle_rate

    if candidate.illegal_rate != best.illegal_rate:
        return candidate.illegal_rate < best.illegal_rate

    return candidate.mean_return > best.mean_return


def train_dqn(config: dict) -> DQNAgent:
    seed_everything(config["seed"])

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------
    cnn_config = config["cnn"]
    dqn_config = config["dqn"]
    training_config = config["training"]

    exploration_config = training_config.get(
        "exploration",
        {"mode": "random"},
    )
    exploration_mode = exploration_config.get("mode", "random")
    exploration_start_episode = exploration_config.get(
        "start_episode",
        0,
    )
    temperature = exploration_config.get("temperature", 0.50)
    resmax_eta = exploration_config.get("eta", 0.50)

    novelty_config = training_config.get("novelty", {})
    novelty_enabled = novelty_config.get("enabled", False)
    decay_power = novelty_config.get("decay_power", 1.0)

    inertia_config = training_config.get("inertia", {})
    inertia_enabled = inertia_config.get("enabled", False)
    inertia_start_episode = inertia_config.get(
        "start_episode",
        0,
    )
    inertia_strength = inertia_config.get("strength", 0.0)

    epsilon_hardset_config = training_config.get(
        "epsilon_hardset",
        {},
    )
    epsilon_hardset_episode = epsilon_hardset_config.get(
        "episode",
    )
    epsilon_hardset_value = epsilon_hardset_config.get(
        "value",
    )

    episodes = int(training_config["episodes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # Agent / environment
    # ---------------------------------------------------------
    agent = DQNAgent(
        config_online=cnn_config,
        config_target=cnn_config,
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
    transfer_config = training_config.get("transfer", {})

    if transfer_config.get("enabled", False):
        print(f"[TRANSFER] loaded backbone from {transfer_config['checkpoint']}")
        transfer_cnn_to_dqn(
            agent, transfer_config["checkpoint"], freeze=transfer_config["freeze"]
        )

    reward_fn = get_reward_fn(
        training_config["reward"],
        gamma=dqn_config["gamma"],
    )

    env = init_world(
        seed=config["seed"],
        max_steps=training_config["max_steps"],
        reward_fn=reward_fn,
        min_solution_steps=training_config.get(
            "min_solution_steps",
            0,
        ),
    )

    metrics = DQNMetrics(rolling_window=10)

    logger = DQNLogger(
        run_dir=config["name"],
        run_config=config,
    )

    validation_interval = VALIDATION_CONFIG["validation_interval"]

    best_validation: EvaluationResult | None = None

    # ---------------------------------------------------------
    # Smoke world generation
    # ---------------------------------------------------------
    for _ in range(20):
        env.reset()

        print(
            "start=",
            env.start_state,
            "goal=",
            env.goal_state,
            "solution_steps=",
            env.solution_steps,
        )

        assert env.solution_steps >= env.min_solution_steps

    try:
        for episode in range(episodes):
            current_episode = episode + 1

            # -------------------------------------------------
            # Episode-level phase changes
            # -------------------------------------------------
            if (
                epsilon_hardset_episode is not None
                and epsilon_hardset_value is not None
                and current_episode == epsilon_hardset_episode
            ):
                agent.epsilon = epsilon_hardset_value

            inertia_active = (
                inertia_enabled and current_episode >= inertia_start_episode
            )

            exploration_active = current_episode >= exploration_start_episode

            # -------------------------------------------------
            # Episode reset
            # -------------------------------------------------
            env.reset()

            previous_action: int | None = None
            agent.q_transform = None

            episode_return = 0.0
            latest_loss = None
            latest_optim_metrics = None
            done = False

            # Episodic familiarity memory.
            episode_counts: dict[
                tuple[tuple[int, int], int],
                int,
            ] = {}

            max_visit_count = 0
            # -------------------------------------------------
            # Environment loop
            # -------------------------------------------------
            while not done:
                state = env.get_state_tensor()
                state_key = tuple(env.state)

                # ---------------------------------------------
                # Runtime Q transform
                # ---------------------------------------------
                if inertia_active:
                    agent.q_transform = make_inertia_transform(
                        previous_action=previous_action,
                        strength=inertia_strength,
                    )
                else:
                    agent.q_transform = None

                # ---------------------------------------------
                # Exploration callback
                # ---------------------------------------------
                def explore(
                    exploration_active=exploration_active,
                    state_key=state_key,
                    episode_counts=episode_counts,
                ) -> int:
                    if not exploration_active:
                        return random.choice([0, 1, 2, 3])

                    if exploration_mode == "boltzmann":
                        return boltzmann_action(
                            agent,
                            env,
                            temperature=temperature,
                            q_transform=agent.q_transform,
                        )

                    if exploration_mode == "resmax":
                        return resmax_action(
                            agent,
                            env,
                            eta=resmax_eta,
                            q_transform=agent.q_transform,
                        )

                    if exploration_mode == "novelty" and novelty_enabled:
                        return novelty_action(
                            env=env,
                            state_key=state_key,
                            visit_counts=episode_counts,
                            decay_power=decay_power,
                        )

                    return random.choice([0, 1, 2, 3])

                # ---------------------------------------------
                # Select action
                # ---------------------------------------------

                action = agent.select_action(
                    env,
                    exploration_fn=explore,
                )

                # ---------------------------------------------
                # Telemetry
                # ---------------------------------------------
                key = (state_key, action)
                visited = episode_counts.get(key, 0)

                episode_counts[key] = visited + 1

                max_visit_count = max(
                    max_visit_count,
                    episode_counts[key],
                )

                # ---------------------------------------------
                # Step environment
                # ---------------------------------------------
                previous_position = tuple(env.state)

                _, reward, done = env.step(action)

                current_position = tuple(env.state)

                # Only real displacement establishes momentum.
                if current_position != previous_position:
                    previous_action = action

                next_state = env.get_state_tensor()

                # ---------------------------------------------
                # Replay + optimization
                # ---------------------------------------------
                agent.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                optimization_result = agent.optimize_model()

                if optimization_result is not None:
                    (
                        latest_loss,
                        latest_optim_metrics,
                    ) = optimization_result

                    logger.log_optimization(latest_optim_metrics)

                agent.update_epsilon()

                episode_return += reward

            # -------------------------------------------------
            # Episode metrics
            # -------------------------------------------------
            success = env.state == env.goal_state

            timeout = env.currentsteps >= env.maxsteps and not success

            episode_metrics = metrics.finish_episode(
                episode=episode,
                episode_return=episode_return,
                steps=env.currentsteps,
                success=success,
                timeout=timeout,
                epsilon=agent.epsilon,
                replay_size=len(agent.replay_buffer),
                latest_loss=latest_loss,
            )

            logger.log_episode(
                episode_metrics,
                latest_optim_metrics,
            )

            logger.writer.add_scalar(
                "agent/exploration_active",
                int(exploration_active),
                current_episode,
            )

            logger.writer.add_scalar(
                "agent/inertia_active",
                int(inertia_active),
                current_episode,
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------
            if current_episode % validation_interval == 0:
                inertia_config = training_config.get("inertia", {})
                inertia_enabled = inertia_config.get(
                    "enabled", False
                ) and current_episode >= inertia_config.get("start_episode", 0)

                validation_result = evaluate_dqn(
                    agent,
                    config={
                        **VALIDATION_CONFIG,
                        "reward": training_config["reward"],
                    },
                    q_transform_fn=(
                        make_inertia_transform if inertia_enabled else None
                    ),
                    inertia_strength=float(inertia_config.get("strength", 0.0)),
                )

                logger.log_validation(
                    episode=current_episode,
                    result=validation_result,
                )

                if is_better(
                    validation_result,
                    best_validation,
                ):
                    best_validation = validation_result

                    logger.save_checkpoint(
                        agent=agent,
                        episode=current_episode,
                        validation=validation_result,
                    )

    finally:
        agent.q_transform = None
        logger.close()

    return agent


def main() -> int:

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
