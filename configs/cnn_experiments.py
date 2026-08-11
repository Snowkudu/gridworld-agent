
from __future__ import annotations
from itertools import product
from typing import Any

ExperimentConfig = dict[str, Any]

BASELINE: list[ExperimentConfig]= [
    {
      "name": "cnn_1ch_baseline",
      "model_type": "cnn",
      "input_ch": 1,
      "conv_channels": [16, 32],
      "kernel_size": 3,
      "padding": 1,
      "pooling": 0,
      "fc_hidden": 128,
      "dropout": 0.0,
      "learning_rate": 0.001,
      "batch_size": 128,
      "weight_decay": 0.0,
      "epochs": 50,
      "patience": 10,
      "min_delta": 0.0,
      "split_seed": 123,
      "experiment_seed": 123
    },
    {
      "name": "cnn_3ch_baseline",
      "model_type": "cnn",
      "input_ch": 3,
      "conv_channels": [16, 32],
      "kernel_size": 3,
      "padding": 1,
      "pooling": 0,
      "fc_hidden": 128,
      "dropout": 0.0,
      "learning_rate": 0.001,
      "batch_size": 128,
      "weight_decay": 0.0,
      "epochs": 50,
      "patience": 10,
      "min_delta": 0.0,
      "split_seed": 123,
      "experiment_seed": 123
    }
  ]

ARCHITECTURE: list[ExperimentConfig]= [
     {
          "name": "cnn_3ch_pool_0",
          "model_type": "cnn",
          "input_ch": 3,
          "conv_channels": [16, 32],
          "kernel_size": 3,
          "padding": 1,
          "pooling": 0,
          "fc_hidden": 128,
          "dropout": 0.0,
          "learning_rate": 0.001,
          "batch_size": 128,
          "weight_decay": 0.0,
          "epochs": 50,
          "patience": 10,
          "min_delta": 0.0,
          "split_seed": 123,
          "experiment_seed": 123
        },
     {
          "name": "cnn_3ch_pool_1",
          "model_type": "cnn",
          "input_ch": 3,
          "conv_channels": [16, 32],
          "kernel_size": 3,
          "padding": 1,
          "pooling": 1,
          "fc_hidden": 128,
          "dropout": 0.0,
          "learning_rate": 0.001,
          "batch_size": 128,
          "weight_decay": 0.0,
          "epochs": 50,
          "patience": 10,
          "min_delta": 0.0,
          "split_seed": 123,
          "experiment_seed": 123
        },
         {
              "name": "cnn_3ch_pool_2",
              "model_type": "cnn",
              "input_ch": 3,
              "conv_channels": [16, 32],
              "kernel_size": 3,
              "padding": 1,
              "pooling": 2,
              "fc_hidden": 128,
              "dropout": 0.0,
              "learning_rate": 0.001,
              "batch_size": 128,
              "weight_decay": 0.0,
              "epochs": 50,
              "patience": 10,
              "min_delta": 0.0,
              "split_seed": 123,
              "experiment_seed": 123
            }
     
]

FINALISTS = [
    # A
    {
        "tag": "fc64_do0_lr1e3_bs16_wd3e4",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "weight_decay": 3e-4,
    },

    # B
    {
        "tag": "fc128_do25_lr5e4_bs32_wd1e3",
        "fc_hidden": 128,
        "dropout": 0.25,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "weight_decay": 1e-3,
    },

    # C
    {
        "tag": "fc64_do0_lr2e3_bs16_wd1e5",
        "fc_hidden": 64,
        "dropout": 0.0,
        "learning_rate": 2e-3,
        "batch_size": 16,
        "weight_decay": 1e-5,
    },

    # D
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
    #271,
   # 389,
   # 467,
   # 593,
   # 719,
   # 839,
    #947,
    #1061,
   # 1223,
]

EXPERIMENT_SEEDS = [
    123,
    211,
    307,
    401,
    503,
   # 601,
   # 701,
   # 809,
   # 907,
  #  1009,
]

WEIGHT_DECAYS = [
    0.0,
    1e-6,
    1e-5,
    1e-4,
    3e-4,
    1e-3,
]


WEIGHT_DECAY_SEEDS = [123, 211, 307, 401, 503]

WEIGHT_DECAY_CONFIGS: list[ExperimentConfig] = [
    {
        **FINAL_CNN_BASE,
        **finalist,
        "name": (
            f"{finalist['tag']}"
            f"_wd{weight_decay:g}"
        ),
        "weight_decay": weight_decay,
        "dataset_seed": 123,
        "experiment_seed": 123,
    }
    for finalist, weight_decay in product(
        FINALISTS,
        WEIGHT_DECAY_SEEDS,
    )
]


FINAL_CONFIGS:list[ExperimentConfig]= [
    {
        **FINAL_CNN_BASE,
        **finalist,
        "name": (
            f"{finalist['tag']}"
            f"_ds{dataset_seed}"
            f"_es{experiment_seed}"
        ),
        "dataset_seed": dataset_seed,
        "experiment_seed": experiment_seed,
    }
    for finalist, dataset_seed, experiment_seed in product(
        FINALISTS,
        DATASET_SEEDS,
        EXPERIMENT_SEEDS,
    )
]


EXPECTED_FINAL_RUNS = (
    len(FINALISTS)
    * len(DATASET_SEEDS)
    * len(EXPERIMENT_SEEDS)
)

assert len(FINAL_CONFIGS) == EXPECTED_FINAL_RUNS

final_names = [config["name"] for config in FINAL_CONFIGS]
assert len(final_names) == len(set(final_names))



SURVIVORS = [

    {
    "tag": "fc128_do0",
    "conv_channels": [64, 64],
    "fc_hidden": 128,
    "dropout": 0.0,
}

]

LEARNING_RATES = [
    5e-4,
    7.5e-4,
    1e-3,
    1.25e-3,
    1.5e-3,
    2e-3,
]

BATCH_SIZES = [
    16,
    32,
    64,
    128,
    256,
]

SWEEP_CONFIG: list[ExperimentConfig] = [
    {
        "name": (
            f"cnn_64_64_{survivor['tag']}"
            f"_lr{lr:g}_bs{batch_size}"
        ),
        "model_type": "cnn",
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "fc_hidden": survivor["fc_hidden"],
        "dropout": survivor["dropout"],
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": 0.0,
        "epochs": 50,
        "patience": 10,
        "min_delta": 0.0,
        "split_seed": 123,
        "experiment_seed": 123,
    }
    for survivor in SURVIVORS
    for lr in LEARNING_RATES
    for batch_size in BATCH_SIZES
]

TRAINING_DEFAULTS: ExperimentConfig = {
    "max_epochs": 200,
    "patience": 10,
    "min_delta": 0.0,
}

GAUNTLET: ExperimentConfig= [
          {
              "name": "cnn_3ch_pool_1",
              "model_type": "cnn",
              "input_ch": 3,
              "conv_channels": [16, 32],
              "kernel_size": 3,
              "padding": 1,
              "pooling": 1,
              "fc_hidden": 128,
              "dropout": 0.0,
              "learning_rate": 0.001,
              "batch_size": 128,
              "weight_decay": 0.0,
              "epochs": 50,
              "patience": 10,
              "min_delta": 0.0,
              "split_seed": 123,
              "experiment_seed": 123
            },
         {
                "name": "cnn_3ch_pool1_k3_p2",
                "model_type": "cnn",
                "input_ch": 3,
                "conv_channels": [16, 32],
                "kernel_size": 3,
                "padding": 2,
                "pooling": 1,
                "fc_hidden": 128,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "batch_size": 128,
                "weight_decay": 0.0,
                "epochs": 50,
                "patience": 10,
                "min_delta": 0.0,
                "split_seed": 123,
                "experiment_seed": 123,
            },
         {
                    "name": "cnn_filters_32_32",
                    "model_type": "cnn",
                    "input_ch": 3,
                    "conv_channels": [32, 32],
                    "kernel_size": 3,
                    "padding": 2,
                    "pooling": 1,
                    "fc_hidden": 128,
                    "dropout": 0.0,
                    "learning_rate": 0.001,
                    "batch_size": 128,
                    "weight_decay": 0.0,
                    "epochs": 50,
                    "patience": 10,
                    "min_delta": 0.0,
                    "split_seed": 123,
                    "experiment_seed": 123,
                }, {
                        "name": "cnn_filters_64_64",
                        "model_type": "cnn",
                        "input_ch": 3,
                        "conv_channels": [64, 64],
                        "kernel_size": 3,
                        "padding": 2,
                        "pooling": 1,
                        "fc_hidden": 128,
                        "dropout": 0.0,
                        "learning_rate": 0.001,
                        "batch_size": 128,
                        "weight_decay": 0.0,
                        "epochs": 50,
                        "patience": 10,
                        "min_delta": 0.0,
                        "split_seed": 123,
                        "experiment_seed": 123,
                    },
             {
                    "name": "cnn_64_64_fc64",
                    "model_type": "cnn",
                    "input_ch": 3,
                    "conv_channels": [64, 64],
                    "kernel_size": 3,
                    "padding": 2,
                    "pooling": 1,
                    "fc_hidden": 64,
                    "dropout": 0.0,
                    "learning_rate": 0.001,
                    "batch_size": 128,
                    "weight_decay": 0.0,
                    "epochs": 50,
                    "patience": 10,
                    "min_delta": 0.0,
                    "split_seed": 123,
                    "experiment_seed": 123,
                },
                {
                    "name": "cnn_64_64_fc128",
                    "model_type": "cnn",
                    "input_ch": 3,
                    "conv_channels": [64, 64],
                    "kernel_size": 3,
                    "padding": 2,
                    "pooling": 1,
                    "fc_hidden": 128,
                    "dropout": 0.0,
                    "learning_rate": 0.001,
                    "batch_size": 128,
                    "weight_decay": 0.0,
                    "epochs": 50,
                    "patience": 10,
                    "min_delta": 0.0,
                    "split_seed": 123,
                    "experiment_seed": 123,
                },
                {
                        "name": "cnn_64_64_fc128_do25",
                        "model_type": "cnn",
                        "input_ch": 3,
                        "conv_channels": [64, 64],
                        "kernel_size": 3,
                        "padding": 2,
                        "pooling": 1,
                        "fc_hidden": 128,
                        "dropout": 0.25,
                        "learning_rate": 0.001,
                        "batch_size": 128,
                        "weight_decay": 0.0,
                        "epochs": 50,
                        "patience": 10,
                        "min_delta": 0.0,
                        "split_seed": 123,
                        "experiment_seed": 123,
                    },
                      {
                            "name": "cnn_64_64_fc64_do0",
                            "model_type": "cnn",
                            "input_ch": 3,
                            "conv_channels": [64, 64],
                            "kernel_size": 3,
                            "padding": 2,
                            "pooling": 1,
                            "fc_hidden": 64,
                            "dropout": 0.0,
                            "learning_rate": 0.001,
                            "batch_size": 128,
                            "weight_decay": 0.0,
                            "epochs": 50,
                            "patience": 10,
                            "min_delta": 0.0,
                            "split_seed": 123,
                            "experiment_seed": 123,
                        },
]
def resolve_config(config: ExperimentConfig) -> ExperimentConfig:
    return {
        **TRAINING_DEFAULTS,
        **config,
    }


def get_experiment_configs(config_set: str) -> list[ExperimentConfig]:
    if config_set == "baseline":
        return [resolve_config(config.copy()) for config in BASELINE]
    if config_set == "sweep":
        return [resolve_config(config.copy()) for config in SWEEP_CONFIG]
    if config_set == "architecture":
            return [resolve_config(config.copy()) for config in ARCHITECTURE]
    if config_set == "finals":
        return [resolve_config(config.copy()) for config in FINAL_CONFIGS]    
    if config_set == "weights":
        return [resolve_config(config.copy()) for config in WEIGHT_DECAY_CONFIGS]    
    raise ValueError(
        f"Unknown config set: {config_set}"
    )
