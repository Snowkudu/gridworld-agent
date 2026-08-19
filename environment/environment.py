from __future__ import annotations

from environment.gridworld import GridWorld
from environment.rewards import manhattan_shaped_reward

# load best mlp model for now
# init a world and feed the state into the mlp
# compare the action to the oracle and record it to another file
# send the output to pygame


def init_world(
    seed: int,
    max_steps: int = 200,
    reward_fn=manhattan_shaped_reward,
    min_solution_steps: int = 0,
) -> GridWorld:

    env = GridWorld(
        10,
        0.3,
        reward_fn,
        max_steps,
        seed,
        min_solution_steps=min_solution_steps,
    )

    env.reset()

    return env
