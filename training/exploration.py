from __future__ import annotations

from collections.abc import Callable

import torch

QTransform = Callable[[torch.Tensor], torch.Tensor]


OPPOSITE_ACTION = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


def make_inertia_transform(
    previous_action: int | None,
    strength: float,
) -> QTransform:
    def transform(q_values: torch.Tensor) -> torch.Tensor:
        if previous_action is None or strength <= 0.0:
            return q_values

        transformed = q_values.clone()

        opposite_action = OPPOSITE_ACTION[previous_action]
        transformed[..., opposite_action] -= strength

        return transformed

    return transform


def resmax_action(
    agent,
    env,
    *,
    eta: float,
    q_transform: QTransform | None = None,
) -> int:
    if eta <= 0.0:
        raise ValueError("eta must be > 0")

    legal_actions = [action for action in range(4) if env.is_legal(action)]

    if not legal_actions:
        raise RuntimeError("No legal actions available")

    with torch.no_grad():
        q_values = agent.get_q_values(env).flatten()
        if q_transform is not None:
            q_values = q_transform(q_values)
        legal_q = q_values[legal_actions]

        q_max = legal_q.max()
        gaps = q_max - legal_q

        weights = 1.0 / (gaps + eta)
        probabilities = weights / weights.sum()

    selected_idx = torch.multinomial(
        probabilities,
        num_samples=1,
    ).item()

    return legal_actions[selected_idx]


def boltzmann_action(
    agent,
    env,
    *,
    temperature: float,
    q_transform: QTransform | None = None,
) -> int:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    legal_actions = [action for action in range(4) if env.is_legal(action)]

    if not legal_actions:
        raise RuntimeError("No legal actions available")

    with torch.no_grad():
        q_values = agent.get_q_values(env)
        if q_transform is not None:
            q_values = q_transform(q_values)
        q_values = q_values.flatten()
        legal_q = q_values[legal_actions]

        probabilities = torch.softmax(
            legal_q / temperature,
            dim=0,
        )

    selected_idx = torch.multinomial(
        probabilities,
        num_samples=1,
    ).item()

    return legal_actions[selected_idx]
