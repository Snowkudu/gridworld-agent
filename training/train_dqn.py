from __future__ import annotations

import random

import numpy as np
import torch

from agents.dqn import DQNAgent
from configs.cnn_experiments import TRAINING_DEFAULTS
from environment.environment import init_world
from environment.evaluate_dqn import EvaluationResult, evaluate_dqn
from models.checkpoint import build_model_from_config
from training.novelty import novelty_action, novelty_base, novelty_sched
from training.train_dqn_metrics import DQNMetrics
from utils.dqn_logging import DQNLogger
from utils.reproducibility import seed_everything
from environment.rewards import (
    get_reward_fn,
    manhattan_shaped_reward,
    sparse_reward,
    REWARD_FUNCTIONS
)
from configs.dqn_experiments import DQN_CONFIG
from training.exploration import boltzmann_action, resmax_action

VALIDATION_CONFIG = {
    "seeds": list(range(20_000, 20_020)),
    "max_steps": 100 ,
    "reward": "sparse",
    "validation_interval": 100
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


def train_dqn(config:dict)->DQNAgent:
    seed_everything(config["seed"])
    cnn_config=config["cnn"]
    dqn_config=config["dqn"]
    training_config=config["training"]
    novelty_config = training_config.get("novelty", {})
    novelty_enabled = novelty_config.get("enabled", False)
    decay_power = novelty_config.get("decay_power", 1.0)
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
    device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
             )
    
    agent = DQNAgent(
        config_online=cnn_config,
        config_target=cnn_config,
        replay_capacity=dqn_config["replay_capacity"],
        batch_size=dqn_config["batch_size"],
        gamma=dqn_config["gamma"],
        learning_rate=dqn_config["learning_rate"],
        epsilon_start=dqn_config["epsilon_start"],
        epsilon_min=dqn_config["epsilon_min"],
        epsilon_decay=dqn_config["epsilon_decay"],
        epsilon_update_interval=dqn_config["epsilon_update_interval"],
        target_sync_interval=dqn_config["target_sync_interval"],
        device=device,
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
    validation_interval=VALIDATION_CONFIG["validation_interval"]
    best_validation: EvaluationResult | None = None
    best_validation_episode: int | None = None

    for _ in range(20):
        env.reset()

        print(
            "start=", env.start_state,
            "goal=", env.goal_state,
            "solution_steps=", env.solution_steps,
        )

        assert env.solution_steps >= env.min_solution_steps

    try:
        episodes = int(training_config["episodes"])

        for episode in range(episodes):
            env.reset()

            episode_return = 0.0
            latest_loss = None
            latest_optim_metrics = None
            done = False

            # Episodic familiarity memory.
            # Counts ALL executed state-action pairs,
            # regardless of whether Q-greedy or novelty exploration chose them.
            episode_counts = {}

            # Telemetry only.
            max_visit_count = 0
            novelty_explores = 0

            while not done:
                state = env.get_state_tensor()
                state_key = tuple(env.state)

                def explore():
                    # episode is zero-indexed internally
                    current_episode = episode + 1

                    if current_episode < exploration_start_episode:
                        return random.choice([0, 1, 2, 3])

                    if exploration_mode == "boltzmann":
                        return boltzmann_action(
                            agent,
                            env,
                            temperature=temperature,
                        )

                    if exploration_mode == "resmax":
                        return resmax_action(
                            agent,
                            env,
                            eta=resmax_eta,
                        )

                    if exploration_mode == "novelty":
                        return novelty_action(
                            env=env,
                            state_key=state_key,
                            visit_counts=episode_counts,
                            decay_power=decay_power,
                        )

                    return random.choice([0, 1, 2, 3])

                action = agent.select_action(
                    env,
                    exploration_fn=explore,
                )


                # Count the action that ACTUALLY happened.
                key = (state_key, action)
                visited = episode_counts.get(key, 0)

                episode_counts[key] = visited + 1
                max_visit_count = max(
                    max_visit_count,
                    episode_counts[key],
                )

                # Environment/task reward only.
                _, reward, done = env.step(action)

                next_state = env.get_state_tensor()

                # No novelty reward enters replay anymore.
                agent.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                optimization_result = agent.optimize_model()

                if optimization_result is not None:
                    latest_loss, latest_optim_metrics = optimization_result
                    logger.log_optimization(latest_optim_metrics)

                agent.update_epsilon()

                # Still the real pot_pos/task reward.
                episode_return += reward

            success = env.state == env.goal_state
            timeout = (
                env.currentsteps >= env.maxsteps
                and not success
            )

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
                    "agent/resmax_active",
                    int((episode + 1) >= exploration_start_episode),
                    episode + 1,
                )
            if (episode + 1) % validation_interval == 0:
                validation_result = evaluate_dqn(
                    agent,
                    config=VALIDATION_CONFIG,
                )
                logger.log_validation(
                    episode=episode + 1,
                    result=validation_result,
                )
                if is_better(
                    validation_result,
                    best_validation,
                ):
                    best_validation = validation_result

                    logger.save_checkpoint(
                        agent=agent,
                        episode=episode + 1,
                        validation=validation_result,
                    )

    finally:
        logger.close()

    return agent

def main() -> int:
    agent=train_dqn(DQN_CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
