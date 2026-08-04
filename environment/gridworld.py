import numpy as np
import torch

from environment.pathfinding import isSolvable
from policies.actions import AGENT, GOAL, OBSTACLE, actions


class GridWorld :
    def __init__(self, grid_size,obstaclesPercent,rewards_fn,maxsteps,seed: int | None=None): #Constructor with randoms start and finish.
        self.reward = 0
        self.grid_size = grid_size
        self.RowMax, self.ColMax = grid_size, grid_size
        self.countobstacles = int( (grid_size* grid_size )*obstaclesPercent)
        self.obstaclesPercent = obstaclesPercent
        self.maxsteps=maxsteps  
        self.currentsteps=0  
        self.reward_fn=rewards_fn
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) 

    def initGrid(self): #Init an empty grid with start and finish.
        self.grid=np.zeros((self.grid_size,self.grid_size))
        self.grid[:]=0
        while(1):
            start=(self.rng.integers(0,self.grid_size), self.rng.integers(0,self.grid_size))
            finish=(self.rng.integers(0,self.grid_size), self.rng.integers(0,self.grid_size))
            if start != finish:
                break
        self.start_state = start
        self.state = self.start_state
        self.goal_state = finish
        self.grid[self.goal_state]=2
        #self.grid[self.start_state]=1
         
    def reset(self): # Place obstacles until we get a solvable grid. Clear grid each attempt.(Start/finish fixed) nyi
        self.initGrid() #init grid with start and finish
        self.state=self.start_state
        attempts = 0
        self.currentsteps=0
        self.reward=0
        while True:
            attempts += 1
            # Clear grid and re-place start/goal
            self.grid = np.zeros((self.grid_size, self.grid_size))
            self.grid[self.goal_state] = 2
            #self.grid[self.start_state] = 1

            # Generate unique obstacle positions that don't overlap start/goal
            obs_set = set()
            while len(obs_set) < self.countobstacles:
                r = self.rng.integers(0, self.grid_size)
                c = self.rng.integers(0, self.grid_size)
                if (r, c) == self.start_state or (r, c) == self.goal_state:
                    continue
                obs_set.add((r, c))

            self.obstacles = np.array(list(obs_set))
            for obs in self.obstacles:
                self.grid[obs[0], obs[1]] = -1  # Obstacle placed

            if isSolvable(self):
                break
            if attempts > 1000:
                raise RuntimeError("Couldn't generate a solvable grid")

        
        attempts = 0 #reset attempts for next time
        return self.state

    def _is_inside_grid(self, position):
        row, col = position
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _verify_action(self, action: int):
        row_change, col_change = actions[action]
        row, col = self.state

        candidate = (
            int(row + row_change),
            int(col + col_change),
        )

        candidate_row, candidate_col = candidate

        if not (
            0 <= candidate_row < self.grid_size
            and 0 <= candidate_col < self.grid_size
        ):
            return tuple(self.state), "illegal_move"

        if self.grid[candidate_row, candidate_col] == OBSTACLE:
            return tuple(self.state), "obstacle_hit"

        if candidate == self.goal_state:
            return candidate, "goal"
        
        return candidate, "moved"

    def step(self, action):#Apply action and return new state, reward, and done flag.
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

    def observation_grid(self) -> np.ndarray: # Returns the grid with the agent and goal positions marked.
        observation = self.grid.copy()
        goal_row, goal_col = map(int, self.goal_state)
        agent_row, agent_col = map(int, self.state)

        observation[goal_row, goal_col] = GOAL
        observation[agent_row, agent_col] = AGENT

        return observation

    def state_vector(self) -> np.ndarray:  #Returns the OBSERVATED GRID (WITH THE AGENT AND GOAL) as a flattened 1D array of float32 values.
        return self.observation_grid().flatten().astype(np.float32)
    
    def get_state_tensor(self) -> torch.Tensor:
         return torch.tensor(
            self.observation_grid(),
            dtype=torch.float32,
        )
    
    def debug_tensor(self):     # Prints the current state of the grid, the agent's position, the goal's position, and some statistics about the state tensor.
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


    
