from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


LOG_DIR = Path("artifacts/p4_cnn/tensorboard")
OUTPUT_DIR = Path("artifacts/p4_cnn/tensorboard_exports")

TAGS = [
    "Accuracy/train",
    "Accuracy/validation",
    "Loss/train",
    "Loss/validation",
    "Performance/epoch_seconds",
    "Performance/time_to_best_seconds",
]

PLOT_TAGS = [
    "Accuracy/train",
    "Accuracy/validation",
    "Loss/train",
    "Loss/validation",
]


def find_run_dirs(log_dir: Path) -> list[Path]:
    event_files = log_dir.rglob("events.out.tfevents*")

    return sorted(
        {event_file.parent for event_file in event_files}
    )


def load_run(run_dir: Path) -> dict[str, list]:
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,
        },
    )

    accumulator.Reload()

    available_tags = set(
        accumulator.Tags().get("scalars", [])
    )

    scalars = {}

    for tag in TAGS:
        if tag in available_tags:
            scalars[tag] = accumulator.Scalars(tag)

    return scalars


def export_all_scalars(
    runs: dict[str, dict[str, list]],
    output_path: Path,
) -> None:
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


def export_summary(
    runs: dict[str, dict[str, list]],
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "run",
                "epochs_logged",
                "best_val_loss",
                "best_val_loss_step",
                "val_accuracy_at_best_loss",
                "max_val_accuracy",
                "final_train_loss",
                "final_train_accuracy",
                "total_epoch_seconds",
                "time_to_best_seconds",
            ]
        )

        for run_name, scalars in runs.items():
            val_loss = scalars.get(
                "Loss/validation",
                [],
            )

            val_accuracy = scalars.get(
                "Accuracy/validation",
                [],
            )

            train_loss = scalars.get(
                "Loss/train",
                [],
            )

            train_accuracy = scalars.get(
                "Accuracy/train",
                [],
            )

            epoch_times = scalars.get(
                "Performance/epoch_seconds",
                [],
            )
            time_to_best = scalars.get(
                "Performance/time_to_best_seconds",
                [],
            )

            if not val_loss:
                continue

            best_loss_event = min(
                val_loss,
                key=lambda event: event.value,
            )

            val_accuracy_by_step = {
                event.step: event.value
                for event in val_accuracy
            }

            val_accuracy_at_best_loss = (
                val_accuracy_by_step.get(
                    best_loss_event.step
                )
            )

            max_val_accuracy = (
                max(
                    event.value
                    for event in val_accuracy
                )
                if val_accuracy
                else None
            )

            final_train_loss = (
                train_loss[-1].value
                if train_loss
                else None
            )

            final_train_accuracy = (
                train_accuracy[-1].value
                if train_accuracy
                else None
            )

            total_epoch_seconds = sum(
                event.value
                for event in epoch_times
            )
            time_to_best_seconds = (
                time_to_best[-1].value
                if time_to_best
                else None
            )

            writer.writerow(
                [
                    run_name,
                    len(val_loss),
                    best_loss_event.value,
                    best_loss_event.step,
                    val_accuracy_at_best_loss,
                    max_val_accuracy,
                    final_train_loss,
                    final_train_accuracy,
                    total_epoch_seconds,
                    time_to_best_seconds,
                ]
            )


def export_plot(
    runs: dict[str, dict[str, list]],
    tag: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    plotted_runs = 0

    for run_name, scalars in runs.items():
        events = scalars.get(tag)

        if not events:
            continue

        steps = [
            event.step
            for event in events
        ]

        values = [
            event.value
            for event in events
        ]

        ax.plot(
            steps,
            values,
            linewidth=1.2,
            alpha=0.8,
            label=run_name,
        )

        plotted_runs += 1

    ax.set_title(
        f"{tag} — {plotted_runs} runs"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(tag)
    ax.grid(alpha=0.2)

    # Legends get monstrous during the gauntlet.
    # Keep it for moderate run counts only.
    if plotted_runs <= 16:
        ax.legend(
            fontsize=7,
            loc="best",
        )

    filename = tag.replace("/", "_")

    fig.tight_layout()

    fig.savefig(
        output_dir / f"{filename}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dirs = find_run_dirs(LOG_DIR)

    print(
        f"Found {len(run_dirs)} TensorBoard run directories"
    )

    runs = {}

    for run_dir in run_dirs:
        run_name = run_dir.relative_to(
            LOG_DIR
        ).as_posix()

        scalars = load_run(run_dir)

        if not scalars:
            continue

        runs[run_name] = scalars

    print(
        f"Loaded {len(runs)} runs containing scalar data"
    )

    export_all_scalars(
        runs,
        OUTPUT_DIR / "all_scalars.csv",
    )

    export_summary(
        runs,
        OUTPUT_DIR / "run_summary.csv",
    )

    for tag in PLOT_TAGS:
        export_plot(
            runs,
            tag,
            OUTPUT_DIR,
        )

    print(
        f"Exports written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()