from __future__ import annotations

import torch
from torch.utils.data import Subset

from data.dataset import split_dataset


def make_dataset(
    sample_count: int = 101,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = torch.arange(
        sample_count * 100,
        dtype=torch.float32,
    ).reshape(sample_count, 10, 10)

    actions = (
        torch.arange(sample_count, dtype=torch.int64) % 4
    )

    return states, actions


def get_indices(split: Subset) -> list[int]:
    return list(split.indices)


def test_split_lengths_use_every_sample() -> None:
    states, actions = make_dataset(sample_count=101)

    splits = split_dataset(states, actions, seed=123)

    assert len(splits.train) == 80
    assert len(splits.val) == 10
    assert len(splits.test) == 11

    assert (
        len(splits.train)
        + len(splits.val)
        + len(splits.test)
        == len(states)
    )


def test_same_seed_produces_identical_splits() -> None:
    states, actions = make_dataset()

    first = split_dataset(states, actions, seed=123)
    second = split_dataset(states, actions, seed=123)

    assert get_indices(first.train) == get_indices(second.train)
    assert get_indices(first.val) == get_indices(second.val)
    assert get_indices(first.test) == get_indices(second.test)


def test_split_indices_do_not_overlap() -> None:
    states, actions = make_dataset()

    splits = split_dataset(states, actions, seed=123)

    train_indices = set(get_indices(splits.train))
    val_indices = set(get_indices(splits.val))
    test_indices = set(get_indices(splits.test))

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    all_indices = train_indices | val_indices | test_indices

    assert all_indices == set(range(len(states)))