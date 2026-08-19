from __future__ import annotations

from itertools import product
from typing import Any

ExperimentConfig = dict[str, Any]
# ---------------------------------------------------------------------
# 1. REPRESENTATION BASELINE
# Same network/training recipe; only representation changes.
# ---------------------------------------------------------------------

BASELINE: list[ExperimentConfig] = [
    {
        "name": "cnn_1ch_baseline",
        "input_ch": 1,
    },
    {
        "name": "cnn_3ch_baseline",
        "input_ch": 3,
    },
]
# ---------------------------------------------------------------------
# ARCHITECTURE
# ---------------------------------------------------------------------
CONV_CHANNELS = [
    (16, 32),
    (32, 64),
    (64, 64),
    (64, 128),
]

POOLING_LEVELS = [0, 1, 2]

KERNEL_PADDING = [
    (3, 1),
    (3, 2),
]


ARCHITECTURE: list[ExperimentConfig] = (
    [
        {
            "tag": f"conv_{c1}_{c2}",
            "conv_channels": [c1, c2],
        }
        for c1, c2 in CONV_CHANNELS
    ]
    + [
        {
            "tag": f"pool{pooling}",
            "conv_channels": [64, 64],
            "pooling": pooling,
        }
        for pooling in POOLING_LEVELS
    ]
    + [
        {
            "tag": f"k{kernel}_p{padding}",
            "conv_channels": [64, 64],
            "kernel_size": kernel,
            "padding": padding,
            "pooling": 1,
        }
        for kernel, padding in KERNEL_PADDING
    ]
)
# ---------------------------------------------------------------------
# 3. CLASSIFICATION HEAD
# ---------------------------------------------------------------------

FC_HIDDEN = [64, 128]
DROPOUTS = [0.0, 0.25]
HEAD: list[ExperimentConfig] = [
    {
        "tag": f"fc{fc_hidden}_do{int(dropout * 100)}",
        "fc_hidden": fc_hidden,
        "dropout": dropout,
    }
    for fc_hidden, dropout in product(FC_HIDDEN, DROPOUTS)
]
# ---------------------------------------------------------------------
# 4. OPTIMIZATION
# ---------------------------------------------------------------------

LEARNING_RATES = [5e-4, 1e-3, 2e-3]
BATCH_SIZES = [16, 32]


OPTIMIZATION: list[ExperimentConfig] = [
    {
        "tag": f"lr{lr:g}_bs{batch_size}",
        "learning_rate": lr,
        "batch_size": batch_size,
    }
    for lr, batch_size in product(LEARNING_RATES, BATCH_SIZES)
]
# ---------------------------------------------------------------------
# 5. WEIGHT DECAY
# ---------------------------------------------------------------------

WEIGHT_DECAYS = [
    ("wd0", 0.0),
    ("wd1e6", 1e-6),
    ("wd1e5", 1e-5),
    ("wd1e4", 1e-4),
    ("wd3e4", 3e-4),
    ("wd1e3", 1e-3),
]


WEIGHT_DECAY: list[ExperimentConfig] = [
    {
        "tag": tag,
        "weight_decay": weight_decay,
    }
    for tag, weight_decay in WEIGHT_DECAYS
]
# ---------------------------------------------------------------------
# 6. FINALISTS — GAUNTLET V2
# ---------------------------------------------------------------------
FINALIST_RECIPES: list[ExperimentConfig] = [
    {
        "tag": "fc64_do0_lr1e3_bs16_wd3e4",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "weight_decay": 3e-4,
    },
    {
        "tag": "fc128_do25_lr5e4_bs32_wd1e3",
        "fc_hidden": 128,
        "dropout": 0.25,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "weight_decay": 1e-3,
    },
    {
        "tag": "fc64_do0_lr2e3_bs16_wd1e5",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 16,
        "weight_decay": 1e-5,
    },
    {
        "tag": "fc128_do0_lr2e3_bs32_wd1e4",
        "fc_hidden": 128,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 32,
        "weight_decay": 1e-4,
    },
]

FINAL_CNN_BASE = {
    "model_type": "cnn",
    "conv_channels": [64, 64],
    "input_ch": 3,
    "kernel_size": 3,
    "padding": 2,
    "pooling": 1,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 0.0,
    "split_seed": 123,
}

DATASET_SEEDS = [
    123,
    271,
    389,
    467,
    593,
    719,
    839,
    947,
    1061,
    1223,
]

EXPERIMENT_SEEDS = [
    123,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
]


FINAL_CONFIGS: list[ExperimentConfig] = [
    {
        **FINAL_CNN_BASE,
        **recipe,
        "name": (f"{recipe['tag']}_ds{dataset_seed}_es{experiment_seed}"),
        "dataset_seed": dataset_seed,
        "experiment_seed": experiment_seed,
    }
    for recipe, dataset_seed, experiment_seed in product(
        FINALIST_RECIPES,
        DATASET_SEEDS,
        EXPERIMENT_SEEDS,
    )
]


EXPECTED_FINAL_RUNS = len(FINALIST_RECIPES) * len(DATASET_SEEDS) * len(EXPERIMENT_SEEDS)
assert len(FINAL_CONFIGS) == EXPECTED_FINAL_RUNS
final_names = [config["name"] for config in FINAL_CONFIGS]
assert len(final_names) == len(set(final_names))


PRE_WD_FINALISTS: list[ExperimentConfig] = [
    {
        "tag": "fc64_do0_lr1e3_bs16",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
    },
    {
        "tag": "fc128_do25_lr5e4_bs32",
        "fc_hidden": 128,
        "dropout": 0.25,
        "learning_rate": 5e-4,
        "batch_size": 32,
    },
    {
        "tag": "fc64_do0_lr2e3_bs16",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 16,
    },
    {
        "tag": "fc128_do0_lr2e3_bs32",
        "fc_hidden": 128,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 32,
    },
]

FINALIST_WEIGHT_DECAY: list[ExperimentConfig] = [
    {
        **finalist,
        "tag": f"{finalist['tag']}_{wd_tag}",
        "weight_decay": weight_decay,
    }
    for finalist, (wd_tag, weight_decay) in product(
        PRE_WD_FINALISTS,
        WEIGHT_DECAYS,
    )
]

assert len(FINALIST_WEIGHT_DECAY) == 24

TRANSFER: list[ExperimentConfig] = [
    {
        "name": "cnn_transfer_128_128_baseline",
        "input_ch": 3,
        "conv_channels": [128, 128],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "fc_hidden": 128,
        "dropout": 0.0,
        # Match baseline training values
        "learning_rate": 1e-3,
        "batch_size": 128,
        "weight_decay": 0.0,
        "max_epochs": 200,
        "patience": 10,
        "min_delta": 0.0,
        "dataset_seed": 123,
        "split_seed": 123,
        "experiment_seed": 123,
    }
]


CONFIG_SETS = {
    "baseline": BASELINE,
    "architecture": ARCHITECTURE,
    "head": HEAD,
    "optimization": OPTIMIZATION,
    "weight_decay": WEIGHT_DECAY,
    "weight_decay_finalists": FINALIST_WEIGHT_DECAY,
    "finals": FINAL_CONFIGS,
    "transfer": TRANSFER,
}


CONFIG_BASES: dict[str, ExperimentConfig] = {
    "baseline": {},
    "architecture": {
        "input_ch": 3,
    },
    "head": {
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
    },
    "optimization": {
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "fc_hidden": 128,
        "dropout": 0.0,
    },
    "weight_decay": {
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "fc_hidden": 128,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 128,
    },
    "weight_decay_finalists": {
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
    },
}


TRAINING_DEFAULTS: ExperimentConfig = {
    "model_type": "cnn",
    "conv_channels": [16, 32],
    "kernel_size": 3,
    "padding": 1,
    "pooling": 0,
    "fc_hidden": 128,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 128,
    "weight_decay": 0.0,
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 0.0,
    "dataset_seed": 123,
    "split_seed": 123,
    "experiment_seed": 123,
}


def get_experiment_configs(
    config_set: str,
) -> list[ExperimentConfig]:
    try:
        configs = CONFIG_SETS[config_set]
    except KeyError:
        raise ValueError(f"Unknown config set: {config_set}") from None

    stage_base = CONFIG_BASES.get(config_set, {})

    return [
        resolve_config(
            {
                **stage_base,
                **config,
            }
        )
        for config in configs
    ]


def resolve_config(config: ExperimentConfig) -> ExperimentConfig:
    resolved = {
        **TRAINING_DEFAULTS,
        **config,
    }

    if "name" not in resolved:
        try:
            resolved["name"] = resolved["tag"]
        except KeyError:
            raise ValueError(
                "Experiment config must define either 'name' or 'tag'"
            ) from None

    return resolved
