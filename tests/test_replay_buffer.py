import random

import pytest
import torch

from agents.replay_buffer import ReplayBuffer, Transition


def make_transition(i: int) -> Transition:
    return Transition(
        state=torch.tensor([float(i)]),
        action=i,
        reward=float(i),
        next_state=torch.tensor([float(i + 1)]),
        done=False,
    )


def test_replay_buffer_starts_empty():
    buffer = ReplayBuffer(capacity=3)

    assert len(buffer) == 0
    assert buffer.capacity == 3


def test_push_adds_transition():
    buffer = ReplayBuffer(capacity=3)
    transition = make_transition(0)

    buffer.push(transition)

    assert len(buffer) == 1
    assert buffer.buffer[0] is transition


def test_push_increases_length_until_capacity():
    buffer = ReplayBuffer(capacity=3)

    buffer.push(make_transition(0))
    buffer.push(make_transition(1))
    buffer.push(make_transition(2))

    assert len(buffer) == 3


def test_buffer_never_exceeds_capacity():
    buffer = ReplayBuffer(capacity=3)

    for i in range(10):
        buffer.push(make_transition(i))

    assert len(buffer) == 3


def test_oldest_transition_is_removed_when_full():
    buffer = ReplayBuffer(capacity=3)

    buffer.push(make_transition(0))
    buffer.push(make_transition(1))
    buffer.push(make_transition(2))
    buffer.push(make_transition(3))

    actions = [transition.action for transition in buffer.buffer]

    assert set(actions) == {1, 2, 3}
    assert 0 not in actions


def test_sample_returns_requested_batch_size():
    buffer = ReplayBuffer(capacity=5)

    for i in range(5):
        buffer.push(make_transition(i))

    batch = buffer.sample(batch_size=3)

    assert len(batch) == 3


def test_sample_returns_transitions():
    buffer = ReplayBuffer(capacity=5)

    for i in range(5):
        buffer.push(make_transition(i))

    batch = buffer.sample(batch_size=3)

    assert all(isinstance(item, Transition) for item in batch)


def test_sample_does_not_return_same_entry_twice():
    buffer = ReplayBuffer(capacity=5)

    for i in range(5):
        buffer.push(make_transition(i))

    random.seed(0)
    batch = buffer.sample(batch_size=3)

    assert len({id(item) for item in batch}) == 3


def test_sample_does_not_modify_buffer():
    buffer = ReplayBuffer(capacity=5)

    for i in range(5):
        buffer.push(make_transition(i))

    size_before = len(buffer)

    buffer.sample(batch_size=3)

    assert len(buffer) == size_before


def test_cannot_sample_more_than_buffer_contains():
    buffer = ReplayBuffer(capacity=5)

    buffer.push(make_transition(0))
    buffer.push(make_transition(1))

    with pytest.raises(ValueError):
        buffer.sample(batch_size=3)
