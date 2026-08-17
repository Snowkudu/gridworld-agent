from copy import deepcopy
from itertools import product
DQN_CONFIG = {
    "name": "dqn_baseline",

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
        "batch_size": 64,
        "gamma": 0.90,
        "learning_rate": 1e-3,

        "epsilon_start": 1.0,
        "epsilon_min": 0.10,
        "epsilon_decay": 0.99995,
        "epsilon_update_interval": 2,

        "target_sync_interval": 100,
    },

    "training": {
        "episodes": 2000,
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 10,

        "novelty": {
            "enabled": False,
            "decay_power": 1.0,
        },
        "exploration": {
    "mode": "random",
}
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

# A — replay/bootstrap warmup test
SPARSE_WARMUP_1000 = make_dqn_config(
    name="dqn_sparse_warmup1000",
    reward="sparse",
    dqn={
        "gamma=0.9"
        "learning_starts": 1000,
    },
)
# B — wider CNN test
SPARSE_CNN_64_64 = make_dqn_config(
    name="dqn_sparse_cnn64x64",
    reward="sparse",
    cnn={
        "conv_channels": (64, 64),
        "fc_hidden": 128,
        "dropout": 0.0,
    },
    dqn={
        "gamma=0.95"
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },
)

DICTATOR_64_64 = make_dqn_config(
    name="dqn_dictator_64x64",
    reward="sparse",
    cnn={
        "conv_channels": (64, 64),
        "fc_hidden": 128,
        "dropout": 0.0,
    },
    dqn={
        "gamma": 0.95,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.9995,
    },
    training={
        "episodes": 100,
        "max_steps": 100,
    },
)
LAST_RUN = make_dqn_config(
    name="dqn_last_run",
    reward="sparse",

    cnn={
        "conv_channels": (64, 64),
        "fc_hidden": 128,
        "dropout": 0.0,
    },

    dqn={
        "gamma": 0.95,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "replay_capacity": 50_000,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9999,
        "target_sync_interval": 60,
    },

    training={
        "episodes": 100,
        "max_steps": 200,
    },
)

REWARD_LABELS = {
    "sparse": "sparse",
    "neutral_step": "neutral",
    "mild_step": "mild",
    "potential_manhattan_zero_terminal": "pot_zero",
    "potential_manhattan_position_terminal": "pot_pos",
}


EPISODES = (1500,2000)
MAX_STEPS = (75,100)


LR_VALUES = (3e-4, 3e-3)

SYNC_VALUES = (50, 200)

NOVELTY_BETAS = (0.05, 0.10, 0.20)

DECAY_POWERS = (0.75, 1.0)


EXPERIMENTS = [

    make_dqn_config(
        name="TEST",
        dqn={
            "gamma": 0.90,
            "epsilon_start": 1.0,
            "epsilon_min": 0.40,
            "epsilon_decay": 0.99995,
            "replay_capacity": 250_000,
        },
        training={
            "episodes": 4000,
            "max_steps": 100,
            "min_solution_steps": 5,
            "reward": "potential_manhattan_position_terminal",
            "exploration": {
                "mode": "boltzmann",
                "start_episode": 550,
                "temperature": 0.50,
            },
        },
    ),
    make_dqn_config(
        name="boltz_250k_rb",
        dqn={
            "gamma": 0.90,
            "epsilon_start": 1.0,
            "epsilon_min": 0.40,
            "epsilon_decay": 0.99995,
            "replay_capacity": 250_000,
        },
        training={
            "episodes": 4000,
            "max_steps": 100,
            "min_solution_steps": 5,
            "reward": "potential_manhattan_position_terminal",
            "exploration": {
                "mode": "boltzmann",
                "start_episode": 550,
                "temperature": 0.50,
            },
        },
    ),
]

EVAL_CONFIG = {
    "seeds": list(range(10_000, 10_050)),
    "max_steps": 200,
    "reward": "sparse",
}