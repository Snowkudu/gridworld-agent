from policies.actions import OBSTACLE

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