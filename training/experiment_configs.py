from __future__ import annotations
from typing import Any
ExperimentConfig= dict[str,Any]

SWEEP_CONFIG :list[ExperimentConfig]=[
    {
        "name": "mlp_256_128_bs32_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Batch-size sweep
    # Everything else matches the winner.
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs16_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 16,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs64_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 64,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs128_lr3e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Dropout sweep
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs32_lr3e-4_do00",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.00,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do05",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.05,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do15",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.15,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-4_do20",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.20,
    },

    # ------------------------------------------------------------------
    # Learning-rate sweep
    # 1e-4 is conservative; 1e-3 and 3e-3 are increasingly aggressive.
    # ------------------------------------------------------------------
    {
        "name": "mlp_256_128_bs32_lr1e-4_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs32_lr1e-3_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_128_bs32_lr3e-3_do10",
        "hidden_sizes": (256, 128),
        "batch_size": 32,
        "learning_rate": 3e-3,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },

    # ------------------------------------------------------------------
    # Capacity sweep
    # ------------------------------------------------------------------
    {
        "name": "mlp_64_64_bs32_lr3e-4_do10",
        "hidden_sizes": (64, 64),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_128_64_bs32_lr3e-4_do10",
        "hidden_sizes": (128, 64),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_256_256_bs32_lr3e-4_do10",
        "hidden_sizes": (256, 256),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
    {
        "name": "mlp_512_256_bs32_lr3e-4_do10",
        "hidden_sizes": (512, 256),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "dropout": 0.10,
    },
]

FINAL_CONFIG : ExperimentConfig={
    
    "name": "mlp_256_128_bs16_lr3e-4_do0.10",
    "hidden_sizes": (256, 128),
    "batch_size": 16,
    "learning_rate": 3e-4,
    "dropout": 0.10,
    "weight_decay": 0.0,
    "max_epochs": 50,
    "patience": 8,
    "min_delta": 1e-4,
    }



def get_experiment_configs(config_set:str)->list[ExperimentConfig]:
    if config_set== "final":
        return [FINAL_CONFIG.copy()]
    if config_set== "sweep":
        return [config.copy() for config in SWEEP_CONFIG ]
    
    raise ValueError("Uknown config, expected sweep or final.")
