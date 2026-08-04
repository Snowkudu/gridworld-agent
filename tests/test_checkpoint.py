from __future__ import annotations
import pytest
import torch
from torch import nn

from models.checkpoint import load_model_from_checkpoint
from models.mlp import MLP


def test_checkpoint_round_trip_reconstructs_identical_model(
    tmp_path,
) -> None:
    torch.manual_seed(123)

    config = {
        "name": "checkpoint_test",
        "hidden_sizes": (256, 128),
        "batch_size": 16,
        "learning_rate": 3e-4,
        "dropout": 0.10,
        "weight_decay": 0.0,
        "max_epochs": 50,
        "patience": 8,
        "min_delta": 1e-4,
    }

    original_model = MLP(
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    )
    original_model.eval()

    example_states = torch.randn(
        8,
        10,
        10,
        dtype=torch.float32,
    )

    with torch.no_grad():
        expected_logits = original_model(example_states)

    checkpoint_path = tmp_path / "test_model.pt"

    torch.save(
        {
            "experiment_name": config["name"],
            "config": config.copy(),
            "model_state_dict": original_model.state_dict(),
        },
        checkpoint_path,
    )

    restored_model, restored_checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        restored_logits = restored_model(example_states)

    assert restored_model.training is False
    assert restored_checkpoint["config"] == config

    torch.testing.assert_close(
        restored_logits,
        expected_logits,
        rtol=0.0,
        atol=0.0,
    )

    dropout_layers = [
        layer
        for layer in restored_model.modules()
        if isinstance(layer, nn.Dropout)
    ]

    assert len(dropout_layers) == 2
    assert all(layer.p == 0.10 for layer in dropout_layers)

def test_checkpoint_requires_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "missing_config.pt"

    torch.save(
        {
            "model_state_dict": MLP().state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="config"):
        load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
        )   

def test_checkpoint_requires_model_state_dict(tmp_path) -> None:
    checkpoint_path = tmp_path / "missing_state_dict.pt"

    torch.save(
        {
            "config": {
                "hidden_sizes": (128, 128),
                "dropout": 0.0,
            },
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="model_state_dict"):
        load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
        )