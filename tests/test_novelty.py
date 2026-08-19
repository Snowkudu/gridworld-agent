import pytest

from training.novelty import novelty_base, novelty_sched


def test_novelty_base_decreases():
    assert novelty_base(0) > novelty_base(1) > novelty_base(10)


def test_novelty_sched_dead_above_threshold():
    assert (
        novelty_sched(
            0.75,
            epsilon_min=0.10,
            epsilon_on=0.50,
            beta_max=0.10,
        )
        == 0.0
    )


def test_novelty_sched_reaches_max_at_epsilon_floor():
    assert novelty_sched(
        0.10,
        epsilon_min=0.10,
        epsilon_on=0.50,
        beta_max=0.10,
    ) == pytest.approx(0.10)
