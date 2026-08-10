from pathlib import Path
from typing import Any

import torch
from torch import nn
from models.cnn import CNN
from models.mlp import MLP


def build_model_from_config(config: dict[str, Any]) -> nn.module:
    type= config.get("model_type","mlp")
    if type== "mlp":
        return MLP(
            input_size=int(config.get("input_size", 100)),
            hidden_sizes=tuple(config["hidden_sizes"]),
            num_actions=int(config.get("num_actions", 4)),
            dropout=float(config.get("dropout", 0.0)),
        )
    elif type == "cnn":
        return CNN(
            input_ch=int(config.get("input_ch", 1)),
            conv_channels=tuple(config.get("conv_channels", (16, 32))),
            kernel_size=int(config.get("kernel_size", 3)),
            padding=int(config.get("padding", 1)),
            pooling=int(config.get("pooling", 0)),
            dropout=float(config.get("dropout", 0.0)),
            fc_hidden=int(config.get("fc_hidden",128))
        )
    raise ValueError(f"Unknown model_type: {type}")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must contain a dictionary.")

    if "config" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'config'.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'."
        )
    model = build_model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
