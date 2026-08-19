# training/run_dqn_transfer.py

from copy import deepcopy

from configs.dqn_experiments import DQN_CONFIG, EVAL_CONFIG
from environment.evaluate_dqn import (
    export_run_tensorboard,
    print_results,
    run_experiment,
)

TRANSFER_CHAMP = deepcopy(DQN_CONFIG)
TRANSFER_CHAMP["name"] = "p6_transfer_champ_same_cnndqn"

TRANSFER_CHAMP["cnn"] = {
    "model_type": "cnn",
    "input_ch": 3,
    "conv_channels": (128, 128),
    "kernel_size": 3,
    "padding": 2,
    "pooling": 1,
    "fc_hidden": 128,
    "dropout": 0.0,
}

TRANSFER_CHAMP["dqn"] = {
    "replay_capacity": 100_000,
    "batch_size": 128,
    "gamma": 0.90,
    "learning_rate": 1e-3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.10,
    "epsilon_decay": 0.9999,
    "epsilon_update_interval": 2,
    "target_sync_interval": 100,
}

TRANSFER_CHAMP["training"] = {
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
}

TRANSFER_CHAMP["transfer"] = {
    "enabled": True,
    "checkpoint": (
        "artifacts/p4_cnn/transfer/cnn_transfer_128_128_baseline/checkpoint.pt"
    ),
}

EXPERIMENTS = [
    TRANSFER_CHAMP,
]


def main():
    results = []
    for config in EXPERIMENTS:
        try:
            result = run_experiment(config=config, eval_config=EVAL_CONFIG)
            results.append(result)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"[FAILED] {config['name']}: {type(exc).__name__}: {exc}")
        finally:
            export_run_tensorboard(config)
    print_results(results)


if __name__ == "__main__":
    main()
