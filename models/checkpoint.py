from pathlib import Path

from models.mlp import MLP
import torch
from typing import Any

def build_model_from_config(
    config: dict[str,Any]        
)->MLP:
    return MLP(
       input_size=int(config.get("input_size", 100)),
        hidden_sizes=tuple(config["hidden_sizes"]),
        num_actions=int(config.get("num_actions", 4)),
        dropout=float(config.get("dropout", 0.0)),
    )

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device : torch.device,
        
)-> tuple[ MLP, dict[str,Any] ]:
    checkpoint_path=Path(checkpoint_path)
    checkpoint=torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(
        f"Checkpoint {checkpoint_path} must contain a dictionary."
    )

    if "config" not in checkpoint:
        raise ValueError(
        f"Checkpoint {checkpoint_path} does not contain 'config'."
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
        f"Checkpoint {checkpoint_path} does not contain "
        "'model_state_dict'."
    )
    model = build_model_from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model , checkpoint
