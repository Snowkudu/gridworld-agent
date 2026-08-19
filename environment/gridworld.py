import numpy as np
import torch

from environment.pathfinding import INF, _bfs_distance_from_goal
from policies.actions import AGENT, GOAL, OBSTACLE, actions


class GridWorld:
    def __init__(
        self,
        grid_size,
        obstaclesPercent,
        rewards_fn,
        maxsteps,
        seed: int | None = None,
        min_solution_steps: int = 0,
    ):  # Constructor with randoms start and finish.
        self.reward = 0.0
        self.grid_size = grid_size
        self.RowMax, self.ColMax = grid_size, grid_size
        self.countobstacles = int((grid_size * grid_size) * obstaclesPercent)
        self.obstaclesPercent = obstaclesPercent
        self.maxsteps = maxsteps
        self.currentsteps = 0
        self.reward_fn = rewards_fn
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.min_solution_steps = min_solution_steps
        self.solution_steps: int | None = None

    def initGrid(self):  # Init an empty grid with start and finish.
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[:] = 0
        while 1:
            start = (
                self.rng.integers(0, self.grid_size),
                self.rng.integers(0, self.grid_size),
            )
            finish = (
                self.rng.integers(0, self.grid_size),
                self.rng.integers(0, self.grid_size),
            )
            if start != finish:
                break
        self.start_state = start
        self.state = self.start_state
        self.goal_state = finish
        self.grid[self.goal_state] = 2
        # self.grid[self.start_state]=1

    def reset(self):
        """
        Generate a solvable world.

        Historical behaviour is preserved when min_solution_steps == 0:
        - choose start/goal
        - reroll obstacles until that pair is solvable

        When a solvable world is shorter than min_solution_steps:
        - reject the start/goal pair
        - generate a new pair
        """

        self.currentsteps = 0
        self.reward = 0
        self.solution_steps = None

        world_attempts = 0

        while True:
            world_attempts += 1

            if world_attempts > 1000:
                raise RuntimeError(
                    "Couldn't generate a valid world with "
                    f"min_solution_steps={self.min_solution_steps}"
                )

            # New start / goal pair.
            self.initGrid()
            self.state = self.start_state

            obstacle_attempts = 0

            while True:
                obstacle_attempts += 1

                if obstacle_attempts > 1000:
                    # This start/goal pair has been sufficiently tortured.
                    # Abandon it and generate a new pair.
                    break

                # Clear world but preserve the current start / goal pair.
                self.grid = np.zeros((self.grid_size, self.grid_size))

                self.grid[self.goal_state] = GOAL

                # Generate unique obstacle positions.
                obs_set = set()

                while len(obs_set) < self.countobstacles:
                    r = self.rng.integers(
                        0,
                        self.grid_size,
                    )
                    c = self.rng.integers(
                        0,
                        self.grid_size,
                    )

                    position = (r, c)

                    if position == self.start_state or position == self.goal_state:
                        continue

                    obs_set.add(position)

                self.obstacles = np.array(list(obs_set))

                for obs in self.obstacles:
                    self.grid[
                        obs[0],
                        obs[1],
                    ] = OBSTACLE

                # One BFS now gives us BOTH:
                # 1. solvability
                # 2. optimal path length
                distances = _bfs_distance_from_goal(self)

                solution_steps = int(distances[self.start_state])

                # Unreachable:
                # keep the SAME start / goal pair
                # and simply reroll obstacles.
                if solution_steps == INF:
                    continue

                # Solvable but too trivial:
                # abandon this start / goal pair entirely.
                if solution_steps < self.min_solution_steps:
                    break

                # Accepted world.
                self.solution_steps = solution_steps
                self.state = self.start_state
                self.currentsteps = 0
                self.reward = 0

                return self.state

    def is_legal(self, action: int):
        tr, tc = self.state
        dr, dc = actions[action]
        row = tr + dr
        col = tc + dc
        bounds = 0 <= row < self.grid_size and 0 <= col < self.grid_size
        if not bounds:
            return False
        collision = self.grid[row, col] == OBSTACLE
        return not collision

    def _verify_action(self, action: int):
        row_change, col_change = actions[action]
        row, col = self.state

        candidate = (
            int(row + row_change),
            int(col + col_change),
        )

        candidate_row, candidate_col = candidate

        if not (
            0 <= candidate_row < self.grid_size and 0 <= candidate_col < self.grid_size
        ):
            return tuple(self.state), "illegal_move"

        if self.grid[candidate_row, candidate_col] == OBSTACLE:
            return tuple(self.state), "obstacle_hit"

        if candidate == self.goal_state:
            return candidate, "goal"

        return candidate, "moved"

    def step(self, action):  # Apply action and return new state, reward, and done flag.
        previous_position = self.state
        self.currentsteps += 1

        new_position, event = self._verify_action(action)

        self.state = new_position
        self.agent_row, self.agent_col = self.state
        if event == "goal":
            done = True
        elif self.currentsteps >= self.maxsteps:
            event = "timeout"
            done = True
        else:
            done = False

        reward = self.reward_fn(
            previous_position,
            new_position,
            self.goal_state,
            event,
        )

        return self.state_vector(), reward, done

    def observation_grid(
        self,
    ) -> np.ndarray:  # Returns the grid with the agent and goal positions marked.
        observation = self.grid.copy()
        goal_row, goal_col = map(int, self.goal_state)
        agent_row, agent_col = map(int, self.state)

        observation[goal_row, goal_col] = GOAL
        observation[agent_row, agent_col] = AGENT

        return observation

    def state_vector(
        self,
    ) -> np.ndarray:  # Returns the OBSERVATED GRID (WITH THE AGENT AND GOAL) as a flattened 1D array of float32 values.
        return self.observation_grid().flatten().astype(np.float32)

    def get_state_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.observation_grid(),
            dtype=torch.float32,
        )

    def debug_tensor(
        self,
    ):  # Prints the current state of the grid, the agent's position, the goal's position, and some statistics about the state tensor.
        np_state = self.observation_grid()
        torch_state = self.get_state_tensor()

        print("Equal:", np.allclose(np_state, torch_state.numpy()))
        print("State:", self.state)
        print("Goal:", self.goal_state)
        print("Agent count:", int((np_state == AGENT).sum()))
        print("Shape:", torch_state.shape)
        print("Min:", torch_state.min().item())
        print("Max:", torch_state.max().item())
        print(torch_state)
