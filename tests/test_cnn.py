import pytest
import torch

from data.representation import to_cnn_1ch, to_cnn_3ch
from models.checkpoint import build_model_from_config, load_model_from_checkpoint
from models.cnn import CNN


def test_to_cnn_1ch():
    x = torch.zeros(4, 100)

    converted = to_cnn_1ch(x)

    assert converted.shape == (4, 1, 10, 10)
    assert converted.dtype == x.dtype


def test_to_cnn_3ch():
    x = torch.zeros(4, 100)

    converted = to_cnn_3ch(x)

    assert converted.shape == (4, 3, 10, 10)


def test_3ch_encoding():
    state = torch.zeros(1, 100)

    state[0, 11] = -1  # obstacle -> row 1, col 1
    state[0, 22] = 1  # agent    -> row 2, col 2
    state[0, 33] = 2  # goal     -> row 3, col 3

    x = to_cnn_3ch(state)

    # Channel 0 = obstacles
    assert x[0, 0, 1, 1] == 1

    # Channel 1 = agent
    assert x[0, 1, 2, 2] == 1

    # Channel 2 = goal
    assert x[0, 2, 3, 3] == 1

    # No channel leakage
    assert x[0, 1, 1, 1] == 0
    assert x[0, 2, 1, 1] == 0

    assert x[0, 0, 2, 2] == 0
    assert x[0, 2, 2, 2] == 0

    assert x[0, 0, 3, 3] == 0
    assert x[0, 1, 3, 3] == 0


def test_3ch_dtype():
    x = torch.zeros(2, 100, dtype=torch.float64)

    converted = to_cnn_3ch(x)

    assert converted.dtype == torch.float64
    assert converted.device == x.device


def test_output_shape():
    model = CNN(
        input_ch=3,
        conv_channels=(64, 64),
        kernel_size=3,
        padding=2,
        pooling=1,
        fc_hidden=128,
    )

    x = torch.zeros(4, 3, 10, 10)

    logits = model(x)

    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_batch_size(batch_size):
    model = CNN(
        input_ch=3,
        conv_channels=(64, 64),
        pooling=1,
        padding=2,
    )

    x = torch.zeros(batch_size, 3, 10, 10)

    logits = model(x)

    assert logits.shape == (batch_size, 4)


@pytest.mark.parametrize("pooling", [0, 1, 2])
def test_pooling(pooling):
    model = CNN(
        input_ch=3,
        conv_channels=(64, 64),
        kernel_size=3,
        padding=2,
        pooling=pooling,
    )

    x = torch.zeros(2, 3, 10, 10)

    logits = model(x)

    assert logits.shape == (2, 4)


def test_rebuild_model():
    config = {
        "model_type": "cnn",
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "dropout": 0.25,
        "fc_hidden": 128,
    }

    model = build_model_from_config(config)

    assert isinstance(model, CNN)
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 64
    assert model.conv2.out_channels == 64
    assert model.fc1.out_features == 128
    assert model.dropout.p == 0.25

    logits = model(torch.zeros(2, 3, 10, 10))

    assert logits.shape == (2, 4)


def test_checkpoint_loading(tmp_path):
    config = {
        "model_type": "cnn",
        "input_ch": 3,
        "conv_channels": [64, 64],
        "kernel_size": 3,
        "padding": 2,
        "pooling": 1,
        "dropout": 0.25,
        "fc_hidden": 128,
    }

    original_model = build_model_from_config(config)
    original_model.eval()

    x = torch.randn(2, 3, 10, 10)

    with torch.no_grad():
        expected_logits = original_model(x)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "config": config,
            "model_state_dict": original_model.state_dict(),
        },
        checkpoint_path,
    )

    restored_model, checkpoint = load_model_from_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    )

    with torch.no_grad():
        restored_logits = restored_model(x)

    assert isinstance(restored_model, CNN)
    assert restored_model.training is False
    assert checkpoint["config"] == config

    torch.testing.assert_close(
        restored_logits,
        expected_logits,
        rtol=0.0,
        atol=0.0,
    )


def test_dropout_after_eval():
    model = CNN(
        input_ch=3,
        conv_channels=(64, 64),
        padding=2,
        pooling=1,
        dropout=0.5,
    )

    model.train()

    assert model.training is True
    assert model.dropout.training is True

    model.eval()

    assert model.training is False
    assert model.dropout.training is False

    x = torch.randn(2, 3, 10, 10)

    with torch.no_grad():
        first = model(x)
        second = model(x)

    torch.testing.assert_close(
        first,
        second,
        rtol=0.0,
        atol=0.0,
    )
