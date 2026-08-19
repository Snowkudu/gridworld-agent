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
    return -1


def neutral_step_reward(
    previous_position,
    current_position,
    goal,
    event,
):
    if event == "goal":
        return 10.0

    if event in ("illegal_move", "obstacle_hit"):
        return -2.0

    if event == "timeout":
        return -5.0

    return 0.0


def mild_step_reward(
    previous_position,
    current_position,
    goal,
    event,
):
    if event == "goal":
        return 10.0

    if event in ("illegal_move", "obstacle_hit"):
        return -2.0

    if event == "timeout":
        return -5.0

    return -0.1


def manhattan_distance(position, goal) -> int:
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def manhattan_shaped_reward(
    previous_position,
    current_position,
    goal,
    event,
):
    if event == "goal":
        return 100.0

    if event == "illegal_move" or event == "obstacle_hit":
        return -2.0

    if event == "timeout":
        return -5.0

    old_distance = manhattan_distance(previous_position, goal)
    new_distance = manhattan_distance(current_position, goal)

    if new_distance < old_distance:
        return -0.5

    return -1.5


def time_distance_reward(
    previous_position,
    current_position,
    goal,
    event,
    *,
    step_cost: float = -0.5,
    distance_weight: float = 0.5,
    illegal_extra: float = -1.5,
    goal_reward: float = 10.0,
    timeout_reward: float = -5.0,
) -> float:
    if event == "goal":
        return goal_reward

    if event == "timeout":
        return timeout_reward

    distance = manhattan_distance(current_position, goal)
    distance_cost = -distance_weight * (distance / 18.0)

    if event in ("illegal_move", "obstacle_hit"):
        return step_cost + illegal_extra + distance_cost

    return step_cost + distance_cost


def time_cost_reward(
    event: str,
    *,
    step_cost: float = -0.5,
    illegal_extra: float = -1.5,
    goal_reward: float = 10.0,
    timeout_reward: float = -5.0,
) -> float:
    if event == "goal":
        return goal_reward

    if event == "timeout":
        return timeout_reward

    if event == "illegal":
        return step_cost + illegal_extra  # -2.0

    # ordinary legal movement
    return step_cost


def make_potential_manhattan_reward(
    gamma: float,
    zero_terminal: bool,
):
    def potential_manhattan_reward(
        previous_position,
        current_position,
        goal,
        event,
    ):
        if event == "goal":
            base_reward = 10.0
        elif event in ("illegal_move", "obstacle_hit"):
            base_reward = -2.0
        elif event == "timeout":
            base_reward = -5.0
        else:
            base_reward = 0.0

        previous_phi = -manhattan_distance(previous_position, goal) / 18.0

        if zero_terminal and event in ("goal", "timeout"):
            current_phi = 0.0
        else:
            current_phi = -manhattan_distance(current_position, goal) / 18.0

        shaping = gamma * current_phi - previous_phi

        return base_reward + shaping

    return potential_manhattan_reward


REWARD_FUNCTIONS = {
    "manhattan": manhattan_shaped_reward,
    "sparse": sparse_reward,
    "neutral_step": neutral_step_reward,
    "mild_step": mild_step_reward,
    "time_distance_reward": time_distance_reward,
    "time_distance": time_distance_reward,
}


def get_reward_fn(
    name: str,
    *,
    gamma: float | None = None,
):
    if name == "potential_manhattan_zero_terminal":
        if gamma is None:
            raise ValueError("potential_manhattan requires gamma")

        return make_potential_manhattan_reward(
            gamma=gamma,
            zero_terminal=True,
        )

    if name == "potential_manhattan_position_terminal":
        if gamma is None:
            raise ValueError("potential_manhattan requires gamma")

        return make_potential_manhattan_reward(
            gamma=gamma,
            zero_terminal=False,
        )

    try:
        return REWARD_FUNCTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown reward function: {name}")
