from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from environment.gridworld import GOAL, OBSTACLE, GridWorld
from environment.rewards import manhattan_shaped_reward
from scripts.verify import DatasetValidationError, validate_arrays

GRID_SIZE = 10
OBSTACLE_COUNT = 30


def make_state(
    *,
    agent: tuple[int, int] = (5, 5),
    goal: tuple[int, int] = (9, 9),
    offset: int = 0,
) -> np.ndarray:
    """
    Construct a valid 10x10 state containing exactly 30 obstacles.

    Obstacles are selected deterministically while excluding the agent, goal,
    and the cell immediately to the agent's right. Action 2 is therefore legal.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    protected = {
        agent,
        goal,
        (agent[0], agent[1] + 1),
    }

    candidates = [
        (row, column)
        for row in range(GRID_SIZE)
        for column in range(GRID_SIZE)
        if (row, column) not in protected
    ]

    rotated = candidates[offset:] + candidates[:offset]

    for row, column in rotated[:OBSTACLE_COUNT]:
        grid[row, column] = -1

    grid[agent] = 1
    grid[goal] = 2

    return grid.reshape(-1)


def make_valid_arrays(
    sample_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(
        [
            make_state(
                agent=(4 + index, 4),
                goal=(9, 9),
                offset=index * 7,
            )
            for index in range(sample_count)
        ]
    ).astype(np.float32)

    # Right: protected by make_state(), therefore legal.
    y = np.full(sample_count, 2, dtype=np.int64)
    return X, y


def test_accepts_valid_dataset() -> None:
    X, y = make_valid_arrays(sample_count=3)

    validate_arrays(X, y)


def test_accepts_dataset_with_fewer_than_200_samples() -> None:
    X, y = make_valid_arrays(sample_count=1)

    validate_arrays(X, y)


def test_rejects_mismatched_state_and_action_counts() -> None:
    X, y = make_valid_arrays(sample_count=2)

    with pytest.raises(
        DatasetValidationError,
        match="State/action count mismatch",
    ):
        validate_arrays(X, y[:1])


@pytest.mark.parametrize(
    ("X_transform", "y_transform", "message"),
    [
        (
            lambda X: X.reshape(1, 10, 10),
            lambda y: y,
            "X must be two-dimensional",
        ),
        (
            lambda X: X,
            lambda y: y.reshape(1, 1),
            "y must be one-dimensional",
        ),
        (
            lambda X: X[:, :-1],
            lambda y: y,
            "X must have shape",
        ),
        (
            lambda X: X.astype(np.float64),
            lambda y: y,
            "X must have dtype",
        ),
        (
            lambda X: X,
            lambda y: y.astype(np.int32),
            "y must have dtype",
        ),
    ],
)
def test_rejects_wrong_shape_or_dtype(
    X_transform,
    y_transform,
    message: str,
) -> None:
    X, y = make_valid_arrays()

    with pytest.raises(DatasetValidationError, match=message):
        validate_arrays(X_transform(X), y_transform(y))


def test_rejects_empty_dataset() -> None:
    X = np.empty((0, 100), dtype=np.float32)
    y = np.empty((0,), dtype=np.int64)

    with pytest.raises(
        DatasetValidationError,
        match="at least one sample",
    ):
        validate_arrays(X, y)


def test_rejects_unknown_state_value() -> None:
    X, y = make_valid_arrays()
    X[0, 99] = 7

    with pytest.raises(
        DatasetValidationError,
        match="invalid state value",
    ):
        validate_arrays(X, y)


@pytest.mark.parametrize("action", [-1, 4, 99])
def test_rejects_action_outside_zero_to_three(action: int) -> None:
    X, y = make_valid_arrays()
    y[0] = action

    with pytest.raises(
        DatasetValidationError,
        match="invalid action",
    ):
        validate_arrays(X, y)


@pytest.mark.parametrize("agent_count", [0, 2])
def test_rejects_incorrect_agent_count(agent_count: int) -> None:
    X, y = make_valid_arrays()
    grid = X[0].reshape(10, 10)

    original_agent = tuple(np.argwhere(grid == 1)[0])
    grid[original_agent] = 0

    if agent_count == 2:
        grid[5, 5] = 1
        grid[6, 5] = 1

    with pytest.raises(
        DatasetValidationError,
        match=rf"contains {agent_count} agents",
    ):
        validate_arrays(X, y)


@pytest.mark.parametrize("goal_count", [0, 2])
def test_rejects_incorrect_goal_count(goal_count: int) -> None:
    X, y = make_valid_arrays()
    grid = X[0].reshape(10, 10)

    original_goal = tuple(np.argwhere(grid == 2)[0])
    grid[original_goal] = 0

    if goal_count == 2:
        grid[8, 8] = 2
        grid[9, 9] = 2

    with pytest.raises(
        DatasetValidationError,
        match=rf"contains {goal_count} goals",
    ):
        validate_arrays(X, y)


@pytest.mark.parametrize("obstacle_count", [29, 31])
def test_rejects_incorrect_obstacle_count(
    obstacle_count: int,
) -> None:
    X, y = make_valid_arrays()
    grid = X[0].reshape(10, 10)

    current_obstacles = list(map(tuple, np.argwhere(grid == -1)))

    if obstacle_count == 29:
        grid[current_obstacles[0]] = 0
    else:
        empty_position = tuple(np.argwhere(grid == 0)[0])
        grid[empty_position] = -1

    with pytest.raises(
        DatasetValidationError,
        match=rf"contains {obstacle_count} obstacles",
    ):
        validate_arrays(X, y)


def test_rejects_action_that_leaves_grid() -> None:
    X = np.stack(
        [
            make_state(
                agent=(0, 5),
                goal=(9, 9),
            )
        ]
    )
    y = np.array([0], dtype=np.int64)  # Up from row zero.

    with pytest.raises(
        DatasetValidationError,
        match="leaves the grid",
    ):
        validate_arrays(X, y)


def test_rejects_action_that_enters_obstacle() -> None:
    X = np.stack(
        [
            make_state(
                agent=(5, 5),
                goal=(9, 9),
            )
        ]
    )
    y = np.array([3], dtype=np.int64)  # 3 = right

    grid = X[0].reshape(10, 10)

    # Preserve the expected obstacle count.
    if grid[5, 6] != -1:
        existing_obstacle = tuple(np.argwhere(grid == -1)[0])
        grid[existing_obstacle] = 0
        grid[5, 6] = -1

    with pytest.raises(
        DatasetValidationError,
        match=r"enters obstacle",
    ):
        validate_arrays(X, y)


def test_rejects_duplicate_states_with_same_label() -> None:
    X, y = make_valid_arrays()
    X = np.concatenate([X, X.copy()])
    y = np.concatenate([y, y.copy()])

    with pytest.raises(
        DatasetValidationError,
        match="Duplicate state detected",
    ):
        validate_arrays(X, y)


def test_rejects_duplicate_states_with_conflicting_labels() -> None:
    X, y = make_valid_arrays()

    X = np.concatenate([X, X.copy()])
    y = np.array([2, 3], dtype=np.int64)

    with pytest.raises(
        DatasetValidationError,
        match="Conflicting labels detected",
    ):
        validate_arrays(X, y)


def test_cli_returns_zero_for_valid_dataset(tmp_path: Path) -> None:
    X, y = make_valid_arrays()
    dataset_path = tmp_path / "valid.npz"

    np.savez_compressed(
        dataset_path,
        states=X,
        actions=y,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify",
            str(dataset_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_cli_returns_nonzero_for_invalid_dataset(
    tmp_path: Path,
) -> None:
    X, y = make_valid_arrays()
    y[0] = 4  # Outside valid action range 0–3.

    dataset_path = tmp_path / "invalid.npz"

    np.savez_compressed(
        dataset_path,
        states=X,
        actions=y,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify",
            str(dataset_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "VALIDATION FAILED" in combined_output


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def make_timeout_test_env(
    *,
    agent: tuple[int, int],
    goal: tuple[int, int],
    obstacles: tuple[tuple[int, int], ...] = (),
    grid_size: int = 4,
    max_steps: int = 1,
) -> GridWorld:
    """Build a deterministic environment one step before timeout."""

    env = GridWorld(
        grid_size,
        0.0,
        manhattan_shaped_reward,
        maxsteps=max_steps,
    )

    env.grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    env.state = agent
    env.start_state = agent
    env.goal_state = goal

    env.grid[goal] = GOAL

    for obstacle in obstacles:
        env.grid[obstacle] = OBSTACLE

    env.agent_row, env.agent_col = agent

    env.maxsteps = max_steps
    env.currentsteps = max_steps - 1

    return env


def test_boundary_hit_on_final_step_times_out() -> None:
    env = make_timeout_test_env(
        agent=(0, 1),
        goal=(3, 3),
    )

    _, reward, done = env.step(UP)

    assert env.currentsteps == env.maxsteps
    assert env.state == (0, 1)
    assert done is True
    assert reward == -5.0


def test_obstacle_hit_on_final_step_times_out() -> None:
    env = make_timeout_test_env(
        agent=(1, 1),
        goal=(3, 3),
        obstacles=((1, 2),),
    )

    _, reward, done = env.step(RIGHT)

    assert env.currentsteps == env.maxsteps
    assert env.state == (1, 1)
    assert done is True
    assert reward == -5.0


def test_legal_final_step_is_applied_before_timeout() -> None:
    env = make_timeout_test_env(
        agent=(1, 1),
        goal=(3, 3),
    )

    state_vector, reward, done = env.step(RIGHT)

    assert env.currentsteps == env.maxsteps
    assert env.state == (1, 2)
    assert done is True
    assert reward == -5.0

    grid = state_vector.reshape(env.grid_size, env.grid_size)

    assert grid[1, 2] == 1
    assert grid[1, 1] == 0
