from __future__ import annotations

import torch

def resmax_action(
    agent,
    env,
    *,
    eta: float,
) -> int:
    if eta <= 0.0:
        raise ValueError("eta must be > 0")

    legal_actions = [
        action
        for action in range(4)
        if env.is_legal(action)
    ]

    if not legal_actions:
        raise RuntimeError("No legal actions available")

    with torch.no_grad():
        q_values = agent.get_q_values(env).flatten()
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
) -> int:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    legal_actions = [
        action
        for action in range(4)
        if env.is_legal(action)
    ]

    if not legal_actions:
        raise RuntimeError("No legal actions available")

    with torch.no_grad():
        
        q_values = agent.get_q_values(env).flatten()
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