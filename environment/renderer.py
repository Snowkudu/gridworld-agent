import environment
from matplotlib.pyplot import grid
import numpy as np , torch
from policies.actions import actions,INF, EMPTY, AGENT, GOAL, OBSTACLE

#def render(s: 'environment.Gridworld'):  # Outputs the grid.
#        grid = np.full((s.grid_size, s.grid_size), "0", dtype=str)
#        for i in range(s.grid_size):
#            for j in range(s.grid_size):
#                if s.grid[i][j] == -1:
#                    grid[i][j] = "-"  # Obstacle
#                elif s.grid[i][j] == 1:
#                    grid[i][j] = "1"  # Start
#                elif s.grid[i][j] == 2:
#                    grid[i][j] = "2"  # Goal
#                else:
#                    grid[i][j] = "0"  # Empty space
#        print("\n".join(" ".join(row) for row in grid),"\n")

def render(self) -> None:
    agent_position = tuple(self.state)
    goal_position = tuple(self.goal_state)

    for row in range(self.grid_size):
        rendered_row = []

        for col in range(self.grid_size):
            position = (row, col)

            if position == agent_position:
                symbol = "A"
            elif position == goal_position:
                symbol = "G"
            elif self.grid[row, col] == OBSTACLE:
                symbol = "#"
            else:
                symbol = "."

            rendered_row.append(symbol)

        print(" ".join(rendered_row))

    print()