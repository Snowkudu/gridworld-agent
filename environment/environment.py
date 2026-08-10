from __future__ import annotations

from environment.gridworld import GridWorld
from environment.rewards import manhattan_shaped_reward

# load best mlp model for now
# init a world and feed the state into the mlp
# compare the action to the oracle and record it to another file
# send the output to pygame


def init_world(seed) -> GridWorld:
    env = GridWorld(10, 0.3, manhattan_shaped_reward, 50, seed)
    env.reset()
    return env


def update_world(env: GridWorld, a: int):

    return env.step(a)


def print(env: GridWorld):
    return env.observation_grid(env)
