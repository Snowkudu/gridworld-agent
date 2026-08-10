
from __future__ import annotations

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
SWEEP_CONFIG: list[ExperimentConfig]= []
TRAINING_DEFAULTS: ExperimentConfig = {
    "max_epochs": 50,
    "patience": 8,
    "min_delta": 0.0,
}
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
        

    raise ValueError(
        f"Unknown config set: {config_set}"
    )
