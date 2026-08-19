import json
from pathlib import Path

WORLD_TIERS = {
    "w0": {
        "seeds": list(range(10_000, 10_500)),
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 0,
    },
    "w5": {
        "seeds": list(range(10_000, 10_500)),
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 5,
    },
    "w10": {
        "seeds": list(range(10_000, 10_500)),
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 10,
    },
    "w15": {
        "seeds": list(range(10_000, 10_500)),
        "max_steps": 100,
        "reward": "potential_manhattan_position_terminal",
        "min_solution_steps": 15,
    },
}


def save_world_tier_results(
    results: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
