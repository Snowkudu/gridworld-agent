from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

from agents.dqn import OptimizationMetrics


@dataclass
class EpisodeMetrics:
    episode: int
    episode_return: float
    steps: int
    success: bool
    timeout: bool
    epsilon: float
    replay_size: int
    latest_loss: float | None

    success_rate: float
    rolling_success: float
    rolling_return: float


class DQNMetrics:
    def __init__(self, rolling_window: int = 10):
        self.total_episodes = 0
        self.total_successes = 0
        self.recent_successes: deque[bool] = deque(maxlen=rolling_window)
        self.recent_returns: deque[float] = deque(maxlen=rolling_window)
        self.window = rolling_window

    def finish_episode(
        self,
        *,
        episode: int,
        episode_return: float,
        steps: int,
        success: bool,
        timeout: bool,
        epsilon: float,
        replay_size: int,
        latest_loss: float | None,
    ) -> EpisodeMetrics:

        self.total_successes += int(success)
        self.recent_successes.append(int(success))
        self.recent_returns.append(episode_return)
        success_rate = self.total_successes / (self.total_episodes + 1)
        rolling_success = sum(self.recent_successes) / len(self.recent_successes)
        rolling_return = sum(self.recent_returns) / len(self.recent_returns)
        self.total_episodes += 1

        return EpisodeMetrics(
            episode=episode,
            episode_return=episode_return,
            steps=steps,
            success=success,
            timeout=timeout,
            epsilon=epsilon,
            replay_size=replay_size,
            latest_loss=latest_loss,
            success_rate=success_rate,
            rolling_success=rolling_success,
            rolling_return=rolling_return,
        )


def metric_as_dict(
    metric: EpisodeMetrics | OptimizationMetrics,
) -> dict[str, object]:
    return asdict(metric)
