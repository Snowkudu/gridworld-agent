from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from policies.actions import AGENT as AGENT_VALUE
from policies.actions import GOAL as GOAL_VALUE
from policies.actions import OBSTACLE as OBSTACLE_VALUE
from policies.actions import actions as ACTION_DELTAS

GRID_SIZE = 10
EXPECTED_FEATURES = GRID_SIZE * GRID_SIZE
EXPECTED_OBSTACLES = 30

STATE_DTYPE = np.dtype(np.float32)
ACTION_DTYPE = np.dtype(np.int64)



ALLOWED_STATE_VALUES = np.array(
    [OBSTACLE_VALUE, 0, AGENT_VALUE, GOAL_VALUE],
    dtype=STATE_DTYPE,
)




class DatasetValidationError(ValueError):
    """Raised when a dataset violates the frozen P1 contract."""


def load_dataset(
    dataset_path: str | Path,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load X and y without silently changing their dtype or shape."""
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise DatasetValidationError(
            f"Dataset does not exist: {dataset_path}"
        )

    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            missing_keys = {"states", "actions"} - set(archive.files)

            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise DatasetValidationError(
                    f"Dataset is missing required array(s): {missing}"
                )

            X = archive["states"]
            y = archive["actions"]

    except DatasetValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(
            f"Could not load dataset {dataset_path}: {exc}"
        ) from exc

    return X, y


def _validate_array_contract(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    if X.ndim != 2:
        raise DatasetValidationError(
            f"X must be two-dimensional; received shape {X.shape}"
        )

    if y.ndim != 1:
        raise DatasetValidationError(
            f"y must be one-dimensional; received shape {y.shape}"
        )

    if X.shape[0] != y.shape[0]:
        raise DatasetValidationError(
            "State/action count mismatch: "
            f"X contains {X.shape[0]} states but y contains "
            f"{y.shape[0]} actions"
        )

    if X.shape[0] == 0:
        raise DatasetValidationError("Dataset must contain at least one sample")

    expected_shape = (X.shape[0], EXPECTED_FEATURES)

    if X.shape != expected_shape:
        raise DatasetValidationError(
            f"X must have shape {expected_shape}; received {X.shape}"
        )

    if X.dtype != STATE_DTYPE:
        raise DatasetValidationError(
            f"X must have dtype {STATE_DTYPE}; received {X.dtype}"
        )

    if y.dtype != ACTION_DTYPE:
        raise DatasetValidationError(
            f"y must have dtype {ACTION_DTYPE}; received {y.dtype}"
        )


def _validate_values(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
) -> None:
    invalid_state_mask = ~np.isin(X, ALLOWED_STATE_VALUES)

    if invalid_state_mask.any():
        sample_index, feature_index = np.argwhere(invalid_state_mask)[0]
        value = X[sample_index, feature_index]

        raise DatasetValidationError(
            "X contains an invalid state value: "
            f"sample={sample_index}, feature={feature_index}, value={value}. "
            f"Allowed values are {ALLOWED_STATE_VALUES.tolist()}"
        )

    invalid_action_mask = ~np.isin(
        y,
        np.fromiter(
            range(len(ACTION_DELTAS)),
            dtype=ACTION_DTYPE,
        )
    )

    if invalid_action_mask.any():
        sample_index = int(np.flatnonzero(invalid_action_mask)[0])

        raise DatasetValidationError(
            "y contains an invalid action: "
            f"sample={sample_index}, action={y[sample_index]}. "
            f"Allowed actions are {sorted(ACTION_DELTAS)}"
        )


def _validate_grid_contents(
    grids: NDArray[np.float32],
) -> None:
    agent_counts = np.count_nonzero(grids == AGENT_VALUE, axis=(1, 2))
    goal_counts = np.count_nonzero(grids == GOAL_VALUE, axis=(1, 2))
    obstacle_counts = np.count_nonzero(
        grids == OBSTACLE_VALUE,
        axis=(1, 2),
    )

    invalid_agent_samples = np.flatnonzero(agent_counts != 1)

    if invalid_agent_samples.size:
        index = int(invalid_agent_samples[0])
        raise DatasetValidationError(
            f"Sample {index} contains {agent_counts[index]} agents; expected 1"
        )

    invalid_goal_samples = np.flatnonzero(goal_counts != 1)

    if invalid_goal_samples.size:
        index = int(invalid_goal_samples[0])
        raise DatasetValidationError(
            f"Sample {index} contains {goal_counts[index]} goals; expected 1"
        )

    invalid_obstacle_samples = np.flatnonzero(
        obstacle_counts != EXPECTED_OBSTACLES
    )

    if invalid_obstacle_samples.size:
        index = int(invalid_obstacle_samples[0])
        raise DatasetValidationError(
            f"Sample {index} contains {obstacle_counts[index]} obstacles; "
            f"expected {EXPECTED_OBSTACLES}"
        )


def _validate_action_legality(
    grids: NDArray[np.float32],
    y: NDArray[np.int64],
) -> None:
    for sample_index, (grid, action) in enumerate(zip(grids, y)):
        agent_positions = np.argwhere(grid == AGENT_VALUE)

        # The one-agent invariant was already checked.
        row, column = agent_positions[0]
        row_delta, column_delta = ACTION_DELTAS[int(action)]

        next_row = int(row + row_delta)
        next_column = int(column + column_delta)

        if not (
            0 <= next_row < GRID_SIZE
            and 0 <= next_column < GRID_SIZE
        ):
            raise DatasetValidationError(
                f"Sample {sample_index} has illegal action {action}: "
                f"move from {(int(row), int(column))} leaves the grid"
            )

        if grid[next_row, next_column] == OBSTACLE_VALUE:
            raise DatasetValidationError(
                f"Sample {sample_index} has illegal action {action}: "
                f"move from {(int(row), int(column))} enters obstacle "
                f"{(next_row, next_column)}"
            )


def _validate_uniqueness(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
) -> None:
    """
    Reject identical flattened states.

    Conflicting labels are checked first so they produce a more useful error
    than the general duplicate-state error.
    """
    states: dict[bytes, tuple[int, int]] = {}

    for sample_index, (state, action) in enumerate(zip(X, y)):
        state_key = state.tobytes()
        action_value = int(action)

        previous = states.get(state_key)

        if previous is None:
            states[state_key] = (sample_index, action_value)
            continue

        previous_index, previous_action = previous

        if previous_action != action_value:
            raise DatasetValidationError(
                "Conflicting labels detected: "
                f"samples {previous_index} and {sample_index} contain the "
                f"same state but actions {previous_action} and {action_value}"
            )

        raise DatasetValidationError(
            "Duplicate state detected: "
            f"samples {previous_index} and {sample_index} are identical"
        )


def validate_arrays(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """Validate in-memory arrays against the frozen P1 dataset contract."""
    _validate_array_contract(X, y)
    _validate_values(X, y)

    grids = X.reshape(-1, GRID_SIZE, GRID_SIZE)

    _validate_grid_contents(grids)
    _validate_action_legality(grids, y)
    _validate_uniqueness(X, y)


def validate_dataset(dataset_path: str | Path) -> None:
    """Load and fully validate a dataset."""
    X, y = load_dataset(dataset_path)
    validate_arrays(X, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a GridWorld dataset against the P1 contract."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the .npz dataset",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        X, y = load_dataset(args.dataset)
        validate_arrays(X, y)
    except DatasetValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        "VALIDATION PASSED: "
        f"{X.shape[0]} samples, state shape={X.shape}, "
        f"state dtype={X.dtype}, action dtype={y.dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())