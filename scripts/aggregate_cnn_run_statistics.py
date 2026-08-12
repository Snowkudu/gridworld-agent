import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(r"^(?P<tag>.+)_ds(?P<dataset_seed>\d+)_es(?P<experiment_seed>\d+)$")


def parse_run_name(run: str) -> tuple[str, int, int]:
    match = RUN_RE.match(run)

    if match is None:
        raise ValueError(f"Invalid run name: {run}")

    return (
        match.group("tag"),
        int(match.group("dataset_seed")),
        int(match.group("experiment_seed")),
    )


def export_catastrophe_summary(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float = 1.0,
) -> None:
    catastrophe_summary = (
        df.groupby("tag")
        .agg(
            runs=("best_val_loss", "size"),
            catastrophes=(
                "best_val_loss",
                lambda values: (values >= threshold).sum(),
            ),
            min_val_loss=("best_val_loss", "min"),
            max_val_loss=("best_val_loss", "max"),
        )
        .reset_index()
    )

    catastrophe_summary["catastrophe_rate"] = (
        catastrophe_summary["catastrophes"] / catastrophe_summary["runs"]
    )

    catastrophe_summary.to_csv(
        output_path,
        index=False,
    )


def export_paired_wins(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []

    tags = sorted(df["tag"].unique())

    for tag_a, tag_b in combinations(tags, 2):
        a = df[df["tag"] == tag_a][
            [
                "dataset_seed",
                "experiment_seed",
                "best_val_loss",
            ]
        ].rename(columns={"best_val_loss": "loss_a"})

        b = df[df["tag"] == tag_b][
            [
                "dataset_seed",
                "experiment_seed",
                "best_val_loss",
            ]
        ].rename(columns={"best_val_loss": "loss_b"})

        paired = a.merge(
            b,
            on=[
                "dataset_seed",
                "experiment_seed",
            ],
            how="inner",
        )

        rows.append(
            {
                "recipe_a": tag_a,
                "recipe_b": tag_b,
                "matched_pairs": len(paired),
                "wins_a": int((paired["loss_a"] < paired["loss_b"]).sum()),
                "wins_b": int((paired["loss_b"] < paired["loss_a"]).sum()),
                "ties": int((paired["loss_a"] == paired["loss_b"]).sum()),
                "mean_delta_a_minus_b": (paired["loss_a"] - paired["loss_b"]).mean(),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    parsed = df["run"].apply(parse_run_name)

    df[["tag", "dataset_seed", "experiment_seed"]] = pd.DataFrame(
        parsed.tolist(),
        index=df.index,
    )

    summary = (
        df.groupby("tag")
        .agg(
            runs=("best_val_loss", "size"),
            mean_val_loss=("best_val_loss", "mean"),
            std_val_loss=("best_val_loss", "std"),
            median_val_loss=("best_val_loss", "median"),
            mean_val_accuracy=("val_accuracy_at_best_loss", "mean"),
            std_val_accuracy=("val_accuracy_at_best_loss", "std"),
            median_val_accuracy=("val_accuracy_at_best_loss", "median"),
            mean_time_to_best=("time_to_best_seconds", "mean"),
            median_time_to_best=("time_to_best_seconds", "median"),
        )
        .sort_values("mean_val_loss")
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )

    export_catastrophe_summary(
        df,
        args.output_dir / "catastrophe_summary.csv",
    )

    export_paired_wins(
        df,
        args.output_dir / "paired_wins.csv",
    )


if __name__ == "__main__":
    main()
