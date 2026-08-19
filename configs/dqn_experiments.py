from copy import deepcopy

DQN_CONFIG = {
    "name": "dqn_champion",
    "seed": 123,
    "cnn": {
        "model_type": "cnn",
        "input_ch": 3,
        "conv_channels": (128, 128),
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "fc_hidden": 128,
        "dropout": 0.0,
    },
    "dqn": {
        "replay_capacity": 100_000,
        "batch_size": 128,
        "gamma": 0.90,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "epsilon_start": 1.0,
        "epsilon_min": 0.10,
        "epsilon_decay": 0.9999,
        "epsilon_update_interval": 2,
        "target_sync_interval": 100,
    },
    "training": {
        "episodes": 2000,
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 5,
        "novelty": {
            "enabled": False,
            "decay_power": 1.0,
        },
        "exploration": {
            "mode": "boltzmann",
            "start_episode": 500,
            "temperature": 0.50,
        },
        "epsilon_hardset": {
            "episode": 1000,
            "value": 0.40,
        },
        "inertia": {
            "enabled": True,
            "start_episode": 1000,
            "strength": 0.75,
        },
    },
}


def make_dqn_config(
    *,
    name: str,
    reward: str | None = None,
    seed: int = 123,
    cnn: dict | None = None,
    dqn: dict | None = None,
    training: dict | None = None,
) -> dict:
    config = deepcopy(DQN_CONFIG)

    config["name"] = name
    config["seed"] = seed

    if reward is not None:
        config["training"]["reward"] = reward

    if cnn:
        config["cnn"].update(cnn)

    if dqn:
        config["dqn"].update(dqn)

    if training:
        config["training"].update(training)

    return config


REWARD_LABELS = {
    "sparse": "sparse",
    "neutral_step": "neutral",
    "mild_step": "mild",
    "potential_manhattan_zero_terminal": "pot_zero",
    "potential_manhattan_position_terminal": "pot_pos",
}

EXPERIMENTS = [
    make_dqn_config(
        name="small_power",
        dqn={
            "target_sync_interval": 50,
        },
        training={
            "episodes": 500,
            "max_steps": 100,
            "min_solution_steps": 5,
            "novelty": {
                "enabled": False,
                "decay_power": 1.0,
            },
            "exploration": {
                "mode": "boltzmann",
                "start_episode": 100,
                "temperature": 1,
            },
            # "epsilon_hardset": {
            #     "episode": 1000,
            #    "value": 0.40,
            # },
            "inertia": {
                "enabled": True,
                "start_episode": 100,
                "strength": 1,
            },
            "transfer": {
                "enabled": True,
                "checkpoint": "artifacts/p4_cnn/cnn_transfer_128_128_baseline.pt",
                "freeze": True,
            },
        },
    )
]

P5_STORY_CONFIGS = [
    make_dqn_config(
        name="p5_story_scratch_champion",
        reward="potential_manhattan_position_terminal",
        seed=123,
        cnn={
            "model_type": "cnn",
            "input_ch": 3,
            "conv_channels": (128, 128),
            "kernel_size": 3,
            "padding": 2,
            "pooling": 1,
            "fc_hidden": 128,
            "dropout": 0.0,
        },
        dqn={
            "replay_capacity": 100_000,
            "batch_size": 128,
            "gamma": 0.90,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_min": 0.10,
            "epsilon_decay": 0.9999,
            "epsilon_update_interval": 2,
            "target_sync_interval": 100,
        },
        training={
            "episodes": 2000,
            "max_steps": 100,
            "min_solution_steps": 5,
            "novelty": {
                "enabled": False,
                "decay_power": 1.0,
            },
            "exploration": {
                "mode": "boltzmann",
                "start_episode": 250,
                "temperature": 0.50,
            },
            "epsilon_hardset": {
                "episode": 500,
                "value": 0.40,
            },
            "inertia": {
                "enabled": True,
                "start_episode": 500,
                "strength": 0.75,
            },
            "transfer": {
                "enabled": False,
            },
        },
    ),
    make_dqn_config(
        name="p5_story_frozen_transfer",
        reward="potential_manhattan_position_terminal",
        seed=123,
        cnn={
            "model_type": "cnn",
            "input_ch": 3,
            "conv_channels": (128, 128),
            "kernel_size": 3,
            "padding": 2,
            "pooling": 1,
            "fc_hidden": 128,
            "dropout": 0.0,
        },
        dqn={
            "replay_capacity": 100_000,
            "batch_size": 128,
            "gamma": 0.90,
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "epsilon_start": 1.0,
            "epsilon_min": 0.10,
            "epsilon_decay": 0.9999,
            "epsilon_update_interval": 2,
            "target_sync_interval": 1000,
        },
        training={
            "episodes": 2000,
            "max_steps": 100,
            "min_solution_steps": 5,
            "novelty": {
                "enabled": False,
                "decay_power": 1.0,
            },
            "exploration": {
                "mode": "boltzmann",
                "start_episode": 100,
                "temperature": 0.50,
            },
            "epsilon_hardset": {
                "episode": 1000,
                "value": 0.40,
            },
            "inertia": {
                "enabled": True,
                "start_episode": 100,
                "strength": 0.75,
            },
            "transfer": {
                "enabled": True,
                "checkpoint": (
                    "artifacts/p4_cnn/transfer/"
                    "cnn_transfer_128_128_baseline/"
                    "checkpoint.pt"
                ),
                "freeze": True,
            },
        },
    ),
]

EVAL_CONFIG = {
    "seeds": list(range(10_000, 10_500)),
    "max_steps": 100,
    "reward": "sparse",
    "min_solution_steps": 5,
}
