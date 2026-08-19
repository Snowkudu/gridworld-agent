# training/novelty.py

import random


def novelty_base(visit_count: int, decay_power: float = 0.5) -> float:
    """Return intrinsic novelty for a state-action pair.

    visit_count is the number of times the pair has already been used
    during the current episode.decay power the intrinsic attractiveness of repeating.
    """
    return 1.0 / ((visit_count + 1) ** decay_power)


def novelty_sched(
    epsilon: float,
    *,
    epsilon_min: float,
    epsilon_on: float,
    beta_max: float,
) -> float:
    """Scale novelty as epsilon-greedy exploration disappears.

    Novelty is:
      - off while epsilon >= epsilon_on
      - increasingly active below epsilon_on
      - beta_max when epsilon reaches epsilon_min
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if epsilon_min <= 0:
        raise ValueError("epsilon_min must be > 0")
    if epsilon_on <= epsilon_min:
        raise ValueError("epsilon_on must be > epsilon_min")
    if beta_max < 0:
        raise ValueError("beta_max must be >= 0")

    if epsilon >= epsilon_on:
        return 0.0

    epsilon = max(epsilon, epsilon_min)

    raw = (1.0 / epsilon) - (1.0 / epsilon_on)
    max_raw = (1.0 / epsilon_min) - (1.0 / epsilon_on)

    return beta_max * (raw / max_raw)


def novelty_action(
    env,
    state_key,
    visit_counts,
    decay_power=1.0,
):
    legal_actions = [action for action in range(4) if env.is_legal(action)]

    weights = [
        novelty_base(
            visit_counts.get((state_key, action), 0),
            decay_power=decay_power,
        )
        for action in legal_actions
    ]

    return random.choices(
        legal_actions,
        weights=weights,
        k=1,
    )[0]
