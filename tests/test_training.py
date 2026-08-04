from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from training.engine import evaluate, train_one_epoch

DEVICE = torch.device("cpu")


def make_loader(
    sample_count: int = 16,
    batch_size: int = 4,
) -> DataLoader:
    states = torch.randn(
        sample_count,
        10,
        10,
        dtype=torch.float32,
    )
    actions = torch.randint(
        low=0,
        high=4,
        size=(sample_count,),
        dtype=torch.int64,
    )

    return DataLoader(
        TensorDataset(states, actions),
        batch_size=batch_size,
        shuffle=False,
    )


def test_train_one_epoch_updates_parameters() -> None:
    torch.manual_seed(123)

    model = MLP(hidden_sizes=(32, 16), dropout=0.0).to(DEVICE)
    loader = make_loader()
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-2)

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    loss, accuracy = train_one_epoch(
        model=model,
        data_loader=loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=DEVICE,
    )

    parameters_after = list(model.parameters())

    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0
    assert model.training is True

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
            strict=True,
        )
    )


def test_evaluate_does_not_change_parameters() -> None:
    torch.manual_seed(123)

    model = MLP(hidden_sizes=(32, 16), dropout=0.10).to(DEVICE)
    loader = make_loader()
    loss_function = nn.CrossEntropyLoss()

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.state_dict().items()
    }

    loss, accuracy = evaluate(
        model=model,
        data_loader=loader,
        loss_function=loss_function,
        device=DEVICE,
    )

    parameters_after = model.state_dict()

    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0
    assert model.training is False

    for name, before in parameters_before.items():
        torch.testing.assert_close(
            parameters_after[name],
            before,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("function_name",),
    [
        ("train",),
        ("evaluate",),
    ],
)
def test_empty_loader_raises_clean_error(
    function_name: str,
) -> None:
    empty_states = torch.empty(
        0,
        10,
        10,
        dtype=torch.float32,
    )
    empty_actions = torch.empty(
        0,
        dtype=torch.int64,
    )

    empty_loader = DataLoader(
        TensorDataset(empty_states, empty_actions),
        batch_size=4,
    )

    model = MLP(hidden_sizes=(32, 16), dropout=0.0).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="No samples"):
        if function_name == "train":
            optimizer = Adam(model.parameters(), lr=1e-3)

            train_one_epoch(
                model=model,
                data_loader=empty_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=DEVICE,
            )
        else:
            evaluate(
                model=model,
                data_loader=empty_loader,
                loss_function=loss_function,
                device=DEVICE,
            )

