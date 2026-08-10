from environment.pathfinding import _bfs_distance_from_goal
from policies.actions import actions


class OraclePolicy:
    def select_action(
        grid,
    ):  # returns the best action to take based on the current state of the grid, closer to the goal.
        dist = _bfs_distance_from_goal(grid=grid)
        agent_row, agent_col = grid.state
        best = None
        temp = dist[agent_row][agent_col]
        best_direction = temp
        for action, (dR, dC) in enumerate(actions):
            newRow, newCol = agent_row + dR, agent_col + dC
            if not (0 <= newRow < grid.RowMax and 0 <= newCol < grid.ColMax):
                continue
            if grid.grid[newRow][newCol] == -1:
                continue
            if dist[newRow][newCol] < best_direction:
                best_direction = dist[newRow][
                    newCol
                ]  # best is used to calc the move that is closer to the goal.
                best = action
        if best is None:
            return 0
        else:
            return best
