from matplotlib.pyplot import grid
import numpy as np , torch
from collections import deque

INF=10**9
actions=[(1, 0), (-1, 0),(0, 1), (0, -1)]

class GridWorld :
    def __init__(self, grid_size,obstaclesPercent): #Constructor with randoms start and finish.
        self.reward = 0
        self.grid_size = grid_size
        self.RowMax, self.ColMax = grid_size, grid_size
        self.countobstacles = int( (grid_size* grid_size )*obstaclesPercent)
        self.maxsteps=grid_size*grid_size
        self.currentsteps=0  

    def initGrid(self): #Init an empty grid with start and finish.
        self.grid=np.zeros((self.grid_size,self.grid_size))
        self.grid[:]=0
        while(1):
            start=(np.random.randint(0,self.grid_size-1), np.random.randint(0,self.grid_size-1))
            finish=(np.random.randint(0,self.grid_size-1), np.random.randint(0,self.grid_size-1))
            if start != finish:
                break
        self.start_state = start
        self.state = self.start_state
        self.goal_state = finish
        self.grid[self.goal_state]=2
        self.grid[self.start_state]=1

    def isSolvable(self):   # Check if the grid is solvable using BFS from start to goal while avoiding obstacles.
                            # Init double ended q with row and col as size then a bfs.
        dist=self._bfs_distance_from_goal()
        agent_row, agent_col = self.state
        
        return dist[agent_row][agent_col] != INF # If distance is INF, it means it's unreachable.
    
    def _bfs_distance_from_goal(self): # Returns an array with distances from the goal.
        tempgrid=self.grid
        dist_array=np.full((self.RowMax,self.ColMax),INF,dtype=np.int32)
        goalR,goalC= self.goal_state
        dist_array[goalR,goalC]=0
        queue= deque([(goalR,goalC)]) #start from goal and Bfs backwards.
        while queue:
            temp_row,temp_col= queue.popleft()
            for itR,itC in actions:
                newRow,newCol= temp_row+itR , temp_col+itC
                if(0 <= newRow < self.RowMax and 0 <= newCol < self.ColMax):
                    if(tempgrid[newRow][newCol] != -1 ):
                       if(dist_array[newRow][newCol] > dist_array[temp_row][temp_col]+1):   
                            dist_array[newRow][newCol] = dist_array[temp_row][temp_col]+1
                            queue.append((newRow,newCol))
        return dist_array               
            
    def reset(self): # Place obstacles until we get a solvable grid. Clear grid each attempt.(Start/finish fixed) nyi
        self.initGrid() #init grid with start and finish
        attempts = 0
        while True:
            attempts += 1
            # Clear grid and re-place start/goal
            self.grid = np.zeros((self.grid_size, self.grid_size))
            self.grid[self.goal_state] = 2
            self.grid[self.start_state] = 1

            # Generate unique obstacle positions that don't overlap start/goal
            obs_set = set()
            while len(obs_set) < self.countobstacles:
                r = np.random.randint(0, self.grid_size)
                c = np.random.randint(0, self.grid_size)
                if (r, c) == self.start_state or (r, c) == self.goal_state:
                    continue
                obs_set.add((r, c))

            self.obstacles = np.array(list(obs_set))
            for obs in self.obstacles:
                self.grid[obs[0], obs[1]] = -1  # Obstacle placed

            if self.isSolvable():
                break
            if attempts > 1000:
                raise RuntimeError("Couldn't generate a solvable grid")

        
        attempts = 0 #reset attempts for next time
        return self.state
    
    def render(self): #Outputs the grid.
        grid = np.full((self.grid_size, self.grid_size), "0", dtype=str)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == -1:
                    grid[i][j] = "-"  # Obstacle
                elif self.grid[i][j] == 1:
                    grid[i][j] = "1"  # Start
                elif self.grid[i][j] == 2:
                    grid[i][j] = "2"  # Goal
                else:
                    grid[i][j] = "0"  # Empty space
        print("\n".join(" ".join(row) for row in grid),"\n")

    def distance(self) -> int: #Manhattan distance from current state to goal.
           
        return abs(self.state[0] - self.goal_state[0]) + abs(self.state[1] - self.goal_state[1])
    
    def step(self, action):#Apply action and return new state, reward, and done flag.
        x, y = self.state
        print("\nCurrent Move:", self.currentsteps, "Out of", self.maxsteps, "\nAction:", action)
        if(self.currentsteps>self.maxsteps):
            print("Max steps reached, reseting.")
            self.state=self.start_state
            done=True
            self.reward+=-5 #penalty for reaching max steps to encourage shorter paths and avoid loops
            return self.state, self.reward,
        else:
            self.currentsteps+=1
            shaping=-0.1 # Default penalty for each step to encourage shorter paths

        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)
        
        temp=self.state
        prevdist = self.distance() #compute prev manhattan dist to apply reward shaping
        self.state = (x, y)
        done = False
        newdist = self.distance() #compute next step distance
        print("Distance: ",prevdist,"And New :",newdist)
        if any((self.obstacles == self.state).all(axis=1)):
            shaping += -2  # Penalty and stay still
            print("Obstacle.")
            self.state = temp
            done = False
        else:
            self.grid[temp] = 0
            self.grid[self.state] = 1 # Set new position according to movement and clear old.
            if(newdist < prevdist):
                shaping += 0.05  # Reward for getting closer
            elif(newdist > prevdist):
                shaping += -0.05  # Penalty for getting farther
        
        if self.state == self.goal_state:
            self.grid[self.state] = 1  
            self.grid[temp] = 0 
            shaping += 10  # Clear old and set finish reward for reaching the goal.
            done = True

        self.reward+=shaping #apply rewards after shaping
        return self.state, self.reward, done
    
    def state_vector(self) -> np.ndarray:
        return np.array(self.state, dtype=np.float32).reshape(-1)  
    
    def get_state_tensor(self) -> torch.Tensor:
        temp=self.state_vector()
        return torch.from_numpy(temp)
    
    def debug_tensor(self):
        np_state = np.array(self.grid, dtype=np.float32)
        torch_state = torch.from_numpy(np_state)

        print("Equal:", np.allclose(np_state, torch_state.numpy()))
        print("Shape:", torch_state.shape)
        print("Min:", torch_state.min().item())
        print("Max:", torch_state.max().item())
        print(torch_state)

    def next_Action(self):
        dist=self._bfs_distance_from_goal()
        agent_row, agent_col = self.state
        best= None
        best_direction= dist[agent_row][agent_col]
        
        for action , (dR,dC) in enumerate(actions):
            newRow, newCol = agent_row + dR, agent_col + dC
            if(0 <= newRow < self.RowMax and 0 <= newCol < self.ColMax):
                if(dist[newRow][newCol] < best_direction and self.grid[newRow][newCol] != -1):
                    best_direction= dist[newRow][newCol] #best is used to calc the move that is closer to the goal.
                    best=action
        if best is None : return 0
        else : return best

    
