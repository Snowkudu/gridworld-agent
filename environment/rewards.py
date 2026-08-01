def sparse_reward(
    previous_position,
    current_position,
    goal,
    event,
):
    if event == "goal":
        return 10.0

    if event == "illegal_move" or event == "obstacle_hit":
        return -2.0
    if event == "timeout":
        return -5
    return -1.0

def manhattan_distance(position, goal) -> int:
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def manhattan_shaped_reward(
    previous_position,
    current_position,
    goal,
    event,
):
    if event == "goal":
        return 10.0

    if event == "illegal_move" or event == "obstacle_hit":
        return -2.0

    if event == "timeout":
        return -5.0

    old_distance = manhattan_distance(previous_position, goal)
    new_distance = manhattan_distance(current_position, goal)

    if new_distance < old_distance:
        return -0.5

    return -1.5