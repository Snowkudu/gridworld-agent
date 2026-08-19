from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

PLOT_EXCLUDE = {
    "train/timeout",
    "target/synched",
}


def find_run_dirs(log_dir: Path) -> list[Path]:
    """Return every directory beneath log_dir containing TensorBoard event files."""
    event_files = log_dir.rglob("events.out.tfevents*")
    return sorted({event_file.parent for event_file in event_files})


def deduplicate_events(events: list) -> list:
    """Keep the newest event for each logged step."""
    latest_by_step = {}

    for event in events:
        current = latest_by_step.get(event.step)

        if current is None or event.wall_time > current.wall_time:
            latest_by_step[event.step] = event

    return [latest_by_step[step] for step in sorted(latest_by_step)]


def load_run(run_dir: Path) -> dict[str, list]:
    """Load every scalar tag present in one TensorBoard event directory."""
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])

    return {tag: deduplicate_events(accumulator.Scalars(tag)) for tag in available_tags}


def export_all_scalars(
    runs: dict[str, dict[str, list]],
    output_path: Path,
) -> None:
    """Export every discovered scalar point into one long-form CSV."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "run",
                "tag",
                "step",
                "wall_time",
                "value",
            ]
        )

        for run_name, scalars in runs.items():
            for tag, events in scalars.items():
                for event in events:
                    writer.writerow(
                        [
                            run_name,
                            tag,
                            event.step,
                            event.wall_time,
                            event.value,
                        ]
                    )


def safe_filename(tag: str) -> str:
    """Convert an arbitrary TensorBoard tag into a Windows-safe filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    return cleaned.strip("_") or "scalar"


def export_plot(
    runs: dict[str, dict[str, list]],
    tag: str,
    output_dir: Path,
) -> None:
    """Plot one discovered scalar tag across every run/series that contains it."""
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_runs = 0

    for run_name, scalars in sorted(runs.items()):
        events = scalars.get(tag)

        if not events:
            continue

        steps = [event.step for event in events]
        values = [event.value for event in events]

        ax.plot(
            steps,
            values,
            linewidth=1.2,
            alpha=0.8,
            label=run_name,
        )

        plotted_runs += 1

    if plotted_runs == 0:
        plt.close(fig)
        return

    ax.set_title(f"{tag} — {plotted_runs} series")
    ax.set_xlabel("Step")
    ax.set_ylabel(tag)
    ax.grid(alpha=0.2)

    # add_scalars() can create several TensorBoard writer directories,
    # and experiment gauntlets can create many more. Avoid unreadable legends.
    if plotted_runs <= 16:
        ax.legend(
            fontsize=7,
            loc="best",
        )

    filename = safe_filename(tag)

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{filename}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def export_all_plots(
    runs: dict[str, dict[str, list]],
    output_dir: Path,
) -> None:
    """Discover every scalar tag and automatically generate its plot."""
    tags = sorted(
        {tag for scalars in runs.values() for tag in scalars if tag not in PLOT_EXCLUDE}
    )

    for tag in tags:
        export_plot(
            runs,
            tag,
            output_dir,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic TensorBoard scalar exporter. "
            "Exports all scalar data to CSV and automatically plots every scalar tag."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="TensorBoard log root to scan recursively.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_dir = args.input
    output_dir = args.input.parent / "exports"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dirs = find_run_dirs(log_dir)

    print(f"Found {len(run_dirs)} TensorBoard event directories")

    runs: dict[str, dict[str, list]] = {}

    for run_dir in run_dirs:
        run_name = run_dir.relative_to(log_dir).as_posix()
        scalars = load_run(run_dir)

        if not scalars:
            continue

        runs[run_name] = scalars

    print(f"Loaded {len(runs)} directories containing scalar data")

    if not runs:
        print("No scalar data found; nothing to export.")
        return

    export_all_scalars(
        runs,
        output_dir / "all_scalars.csv",
    )

    export_all_plots(
        runs,
        output_dir,
    )

    print(f"Exports written to: {output_dir}")


if __name__ == "__main__":
    main()
