import pytest

from agents.dqn import OptimizationMetrics
from training.train_dqn_metrics import (
    DQNMetrics,
    EpisodeMetrics,
    metric_as_dict,
)


def test_finish_episode_first_episode():
    metrics = DQNMetrics(rolling_window=10)

    result = metrics.finish_episode(
        episode=0,
        episode_return=-25.0,
        steps=20,
        success=True,
        timeout=False,
        epsilon=0.9,
        replay_size=20,
        latest_loss=None,
    )

    assert isinstance(result, EpisodeMetrics)

    assert result.episode == 0
    assert result.episode_return == -25.0
    assert result.steps == 20
    assert result.success is True
    assert result.timeout is False
    assert result.epsilon == 0.9
    assert result.replay_size == 20
    assert result.latest_loss is None

    assert result.success_rate == pytest.approx(1.0)
    assert result.rolling_success == pytest.approx(1.0)
    assert result.rolling_return == pytest.approx(-25.0)


def test_finish_episode_tracks_cumulative_and_rolling_metrics():
    metrics = DQNMetrics(rolling_window=2)

    metrics.finish_episode(
        episode=0,
        episode_return=10.0,
        steps=10,
        success=True,
        timeout=False,
        epsilon=1.0,
        replay_size=10,
        latest_loss=None,
    )

    second = metrics.finish_episode(
        episode=1,
        episode_return=-20.0,
        steps=20,
        success=False,
        timeout=True,
        epsilon=0.9,
        replay_size=30,
        latest_loss=0.5,
    )

    assert second.success_rate == pytest.approx(0.5)
    assert second.rolling_success == pytest.approx(0.5)
    assert second.rolling_return == pytest.approx(-5.0)

    third = metrics.finish_episode(
        episode=2,
        episode_return=-30.0,
        steps=30,
        success=False,
        timeout=True,
        epsilon=0.8,
        replay_size=60,
        latest_loss=0.25,
    )

    # cumulative history: [success, failure, failure]
    assert third.success_rate == pytest.approx(1 / 3)

    # rolling window=2 should now contain only episodes 1 and 2
    assert third.rolling_success == pytest.approx(0.0)
    assert third.rolling_return == pytest.approx(-25.0)


def test_metric_as_dict_converts_episode_metrics():
    episode_metrics = EpisodeMetrics(
        episode=4,
        episode_return=-100.0,
        steps=80,
        success=False,
        timeout=True,
        epsilon=0.5,
        replay_size=500,
        latest_loss=0.1,
        success_rate=0.4,
        rolling_success=0.3,
        rolling_return=-120.0,
    )

    result = metric_as_dict(episode_metrics)

    assert result == {
        "episode": 4,
        "episode_return": -100.0,
        "steps": 80,
        "success": False,
        "timeout": True,
        "epsilon": 0.5,
        "replay_size": 500,
        "latest_loss": 0.1,
        "success_rate": 0.4,
        "rolling_success": 0.3,
        "rolling_return": -120.0,
    }


def test_metric_as_dict_converts_optimization_metrics():
    optimization_metrics = OptimizationMetrics(
        optimization_step=100,
        td_loss=0.25,
        predicted_q_mean=4.0,
        bellman_target_mean=4.5,
        target_next_q_mean=5.0,
        target_synced=True,
        parameter_gap=0.0,
    )

    result = metric_as_dict(optimization_metrics)

    assert result == {
        "optimization_step": 100,
        "td_loss": 0.25,
        "predicted_q_mean": 4.0,
        "bellman_target_mean": 4.5,
        "target_next_q_mean": 5.0,
        "target_synced": True,
        "parameter_gap": 0.0,
    }


def test_finish_episode_preserves_latest_loss():
    metrics = DQNMetrics(rolling_window=10)

    result = metrics.finish_episode(
        episode=0,
        episode_return=-50.0,
        steps=40,
        success=False,
        timeout=False,
        epsilon=0.75,
        replay_size=200,
        latest_loss=0.123,
    )

    assert result.latest_loss == pytest.approx(0.123)
