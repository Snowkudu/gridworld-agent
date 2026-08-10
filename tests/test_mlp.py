import pytest
import torch
from torch import nn

from models.mlp import MLP


def test_mlp():
    model = MLP()
    example_states = torch.zeros(size=(32, 100), dtype=torch.float32)
    logits = model(example_states)
    assert logits.shape == (32, 4)
    assert logits.dtype == torch.float32


def test_mlp_accepts_grids():
    model = MLP()
    example_states = torch.zeros(size=(8, 10, 10), dtype=torch.float32)
    logits = model(example_states)
    assert logits.shape == (8, 4)


@pytest.mark.parametrize(
    "dropout",
    [-0.1, 1.0, 1.1],
)
def test_mlp_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        MLP(dropout=dropout)


def test_mlp_adds_dropout_after_each_hidden_layer() -> None:
    model = MLP(
        hidden_sizes=(64, 32, 16),
        dropout=0.25,
    )

    dropout_layers = [
        module for module in model.modules() if isinstance(module, nn.Dropout)
    ]

    assert len(dropout_layers) == 3
    assert all(layer.p == 0.25 for layer in dropout_layers)


def test_mlp_custom_architecture_output_shape() -> None:
    model = MLP(
        input_size=100,
        hidden_sizes=(256, 128),
        num_actions=4,
        dropout=0.10,
    )

    states = torch.zeros(
        7,
        10,
        10,
        dtype=torch.float32,
    )

    logits = model(states)

    assert logits.shape == (7, 4)
    assert logits.dtype == torch.float32
