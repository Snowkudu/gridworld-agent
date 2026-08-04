from collections import deque 
import numpy as np
from policies.actions import INF, actions


def isSolvable(grid):   # Check if the grid is solvable using BFS from start to goal while avoiding obstacles.
                            # Init double ended q with row and col as size then a bfs.
        dist=_bfs_distance_from_goal(grid=grid)
        agent_row, agent_col = grid.state
        
        return dist[agent_row][agent_col] != INF # If distance is INF, it means it's unreachable.

def _bfs_distance_from_goal(grid) -> np.ndarray: # Returns an array with distances from the goal.
        tempgrid=grid.grid
        dist_array=np.full((grid.RowMax,grid.ColMax),INF,dtype=np.int32)
        goalR,goalC= grid.goal_state
        dist_array[goalR,goalC]=0
        queue= deque([(goalR,goalC)]) #start from goal and Bfs backwards.
        while queue:
            temp_row,temp_col= queue.popleft()
            for itR,itC in actions:
                newRow,newCol= temp_row+itR , temp_col+itC
                if(0 <= newRow < grid.RowMax and 0 <= newCol < grid.ColMax):
                    if(tempgrid[newRow][newCol] != -1 ):
                       if(dist_array[newRow][newCol] > dist_array[temp_row][temp_col]+1):   
                            dist_array[newRow][newCol] = dist_array[temp_row][temp_col]+1
                            queue.append((newRow,newCol))
        return dist_array               