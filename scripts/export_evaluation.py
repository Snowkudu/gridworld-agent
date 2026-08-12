import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as file:
        data = json.load(file)

    episodes = pd.DataFrame(data["episodes"])

    summary = pd.DataFrame(
        [
            {
                "episodes": len(episodes),
                "success_rate": episodes["success"].mean(),
                "timeout_rate": episodes["timeout"].mean(),
                "mean_steps": episodes["steps"].mean(),
                "median_steps": episodes["steps"].median(),
                "mean_oracle_agreement": episodes["oracle_agreement"].mean(),
                "mean_repeated_states": episodes["repeated_states"].mean(),
                "mean_max_state_visits": episodes["max_state_visits"].mean(),
            }
        ]
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.output_dir / "evaluation_summary.csv", index=False)
    summary.to_csv(args.output_dir / "rollout_summary.csv", index=False)


if __name__ == "__main__":
    main()
