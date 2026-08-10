from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.dataset import extract_dataset_torch, split_dataset
from models.checkpoint import load_model_from_checkpoint
from models.mlp import MLP
from training.engine import (
    collect_action_diagnostics,
    evaluate,
    train_one_epoch,
)
from training.experiment_configs import get_experiment_configs
from training.metrics import (
    calculate_majority_baseline,
    save_json,
)
from utils.reproducibility import create_generator, seed_everything


def save_selected_model_diagnostics(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint: dict,
    majority_baseline: dict,
    root: Path,
) -> None:

    test_diagnostics = collect_action_diagnostics(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    for action, accuracy in enumerate(test_diagnostics["per_action_accuracy"]):
        true_count = test_diagnostics["true_action_counts"][action]
        predicted_count = test_diagnostics["predicted_action_counts"][action]
        accuracy_text = "N/A" if accuracy is None else f"{accuracy:.2%}"
        print(
            f"Action {action}: "
            f"accuracy={accuracy_text}, "
            f"true samples={true_count}, "
            f"predicted={predicted_count}"
        )
    print("\nConfusion matrix: rows=true, columns=predicted")

    for row in test_diagnostics["confusion_matrix"]:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate supervised MLP models."
    )
    parser.add_argument(
        "--config-set",
        choices=("final", "sweep"),
        default="final",
        help=("Run complete model tests or only the best."),
    )
    return parser.parse_args()


def main() -> int: 
    dataset_path = (
        Path(__file__).parents[1]
        / "data"
        / "raw"
        / "gridworld_2000ep_200ms_123seed.npz"
    )
    args = parse_args()
    experiment_configs = get_experiment_configs(args.config_set)
    split_seed = 123
    experiment_seed = 123

    artifact_root = Path("artifacts") / "p2_mlp"
    artifact_root.mkdir(parents=True, exist_ok=True)

    tensor_X, tensor_y = extract_dataset_torch(dataset_path)

    splits = split_dataset(tensor_X, tensor_y, seed=123)

    majority_baseline = calculate_majority_baseline(
        labels=tensor_y,
        train_subset=splits.train,
        validation_subset=splits.val,
        test_subset=splits.test,
    )
    save_json(
        artifact_root / "majority_baseline.json",
        {
            "split_seed": 123,
            **majority_baseline,
        },
    )
    print(
        f"Traning action counts: | "
        f"{majority_baseline['training_action_counts']} | "
        f"Majority action : "
        f"{majority_baseline['majority_action']}"
        f"Test accuracy : "
        f"{majority_baseline['test_accuracy']:.2%}"
    )  # majority action prints

    train_loader = DataLoader(splits.train, batch_size=128, shuffle=True)
    val_loader = DataLoader(splits.val, batch_size=128, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=128, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # use gpu if available, otherwise use cpu

    loss_function = nn.CrossEntropyLoss()
    epochs = 50

    checkpoint_dir = Path("data/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_results = []

    for config in experiment_configs:
        seed_everything(experiment_seed)
        train_generator = create_generator(experiment_seed)

        experiment_name = config["name"]
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0

        max_epochs = int(config["max_epochs"])
        patience = int(config["patience"])
        min_delta = float(config["min_delta"])

        run_dir = artifact_root / experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "metrics.json"

        epoch_history: list[dict[str, object]] = []

        print("\n" + "=" * 80)
        print(f"Starting experiment: {experiment_name}")
        print(f"Hidden sizes:         {config['hidden_sizes']}")
        print(f"Batch size:           {config['batch_size']}")
        print(f"Learning rate:        {config['learning_rate']}")
        print(f"Weight decay:         {config['weight_decay']}")
        print("=" * 80)

        model = MLP(hidden_sizes=config["hidden_sizes"], dropout=config["dropout"]).to(
            device
        )

        optimizer = Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        train_loader = DataLoader(
            splits.train,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=train_generator,
        )

        for epoch in range(1, max_epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                data_loader=train_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=device,
            )

            validation_loss, validation_accuracy = evaluate(
                model=model,
                data_loader=val_loader,
                loss_function=loss_function,
                device=device,
            )
            epoch_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )

            if validation_loss < best_val_loss - min_delta:
                best_val_loss = validation_loss
                best_val_accuracy = validation_accuracy
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "experiment_name": experiment_name,
                        "config": config.copy(),
                        "split_seed": split_seed,
                        "experiment_seed": experiment_seed,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_accuracy": train_accuracy,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping {experiment_name} at epoch {epoch}")
                break

            print(
                f"{experiment_name} | "
                f"epoch {epoch:02d}/{epochs} | "
                f"train loss={train_loss:.4f}, "
                f"train acc={train_accuracy:.2%} | "
                f"val loss={validation_loss:.4f}, "
                f"val acc={validation_accuracy:.2%}"
            )

        save_json(
            metrics_path,
            {
                "experiment_name": experiment_name,
                "config": config,
                "split_seed": split_seed,
                "experiment_seed": experiment_seed,
                "epochs_completed": len(epoch_history),
                "early_stopped": len(epoch_history) < epochs,
                "patience": patience,
                "min_delta": min_delta,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "best_validation_accuracy": best_val_accuracy,
                "history": epoch_history,
            },
        )  # save the metrics into json after each epoch

        print(
            f"\nCompleted {experiment_name}: "
            f"best epoch={best_epoch}, "
            f"best val loss={best_val_loss:.4f}, "
            f"best val acc={best_val_accuracy:.2%}"
        )
        experiment_results.append(
            {
                "name": experiment_name,
                "checkpoint_path": checkpoint_path,
                "best_epoch": best_epoch,
                "best_validation_loss": best_val_loss,
                "best_validation_accuracy": best_val_accuracy,
            }
        )

    best_result = min(
        experiment_results,
        key=lambda result: result["best_validation_loss"],
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for result in experiment_results:
        print(
            f"{result['name']:<26} | "
            f"epoch={result['best_epoch']:02d} | "
            f"val loss={result['best_validation_loss']:.4f} | "
            f"val acc={result['best_validation_accuracy']:.2%}"
        )

    print("\nSelected model:")
    print(f"Name:            {best_result['name']}")
    print(f"Best epoch:      {best_result['best_epoch']}")
    print(f"Validation loss: {best_result['best_validation_loss']:.4f}")
    print(f"Validation acc:  {best_result['best_validation_accuracy']:.2%}")

    checkpoint = torch.load(best_result["checkpoint_path"], map_location=device)

    best_model, best_checkpoint = load_model_from_checkpoint(
        checkpoint_path=best_result["checkpoint_path"],
        device=device,
    )

    bestLoss, bestAcc = evaluate(
        model=best_model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )
    save_selected_model_diagnostics(
        model=best_model,
        test_loader=test_loader,
        device=device,
        checkpoint=checkpoint,
        majority_baseline=majority_baseline,
        root=artifact_root,
    )
    save_path = Path("results")
    save_json(
        (save_path / "mlp_final_results.json"),
        {
            "phase": "P2",
            "dataset": dataset_path.name,
            "split_seed": best_checkpoint["split_seed"],
            "experiment_seed": best_checkpoint["experiment_seed"],
            "selected_config": best_checkpoint["config"],
            "best_epoch": best_checkpoint["epoch"],
            "validation_loss": best_checkpoint["validation_loss"],
            "validation_accuracy": best_checkpoint["validation_accuracy"],
            "test_loss": bestLoss,
            "test_accuracy": bestAcc,
        },
    )  # save the berst metrics into the results folder

    print(f"Best model was from model {checkpoint['experiment_name']}")
    print(f"Test loss: {bestLoss:.4f}")
    print(f"Test accuracy: {bestAcc:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
