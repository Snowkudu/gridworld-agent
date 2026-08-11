from __future__ import annotations
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from configs.cnn_experiments import get_experiment_configs
from data.dataset import extract_dataset_torch, split_dataset
from data.representation import to_cnn_1ch, to_cnn_3ch
from environment.environment import init_world
from models.checkpoint import load_model_from_checkpoint
from models.cnn import CNN
from training.engine import evaluate, train_one_epoch
from training.metrics import save_json
from utils.reproducibility import create_generator, seed_everything
import time
from torch.utils.tensorboard import SummaryWriter


from pathlib import Path

from configs.cnn_experiments import DATASET_SEEDS
from scripts.generate_dataset import generate_dataset  # use your actual generator import


DATA_DIR = Path("data/raw")


def ensure_final_datasets() -> None:
    for seed in DATASET_SEEDS:
        dataset_path = (
            DATA_DIR
            / f"gridworld_2000ep_200ms_{seed}seed.npz"
        )

        if dataset_path.exists():
            print(f"Dataset exists: {dataset_path}")
            continue
        
        print(f"Generating dataset seed={seed}")
        env= init_world(seed=seed,max_steps=200)
        generate_dataset(
            env=env,
            episodes=2000,
            out_path=dataset_path,
            seed=seed,
        )




def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-set",
        choices=("baseline", "architecture","sweep","finals","weights"),
        default="baseline",
    )
    return parser.parse_args()




def main():

    args=parse_args()
    experiment_configs= get_experiment_configs(args.config_set)
    epochs=50
    epoch_history: list[dict[str, object]] = []

    experiment_name="cnn_first_try"
    artifact_root = Path("artifacts") / "p4_cnn"
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifact_root / experiment_name
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    # epochs_without_improvement=best_epoch=0
    # patience=10
    #min_delta=0.0
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    experiment_results = []
    
    

    
   
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = nn.CrossEntropyLoss()
    
    for config in experiment_configs:

        input_ch = int(config["input_ch"])
        batch_size = int(config["batch_size"])
        max_epochs = int(config["max_epochs"])
        conv_channels=  tuple(config["conv_channels"])
        kernel_size= int(config["kernel_size"])
        fc_hidden=int(config["fc_hidden"])
        dropout=float(config["dropout"])
        patience = int(config["patience"])
        min_delta = float(config["min_delta"])
        pooling = int(config["pooling"])
        padding = int(config["padding"])
        dataset_seed= int(config["dataset_seed"])
        split_seed= int(config["split_seed"])
        experiment_seed= int(config["experiment_seed"])

        experiment_name = config["name"]
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0
        dataset_path = (
            Path("data/raw")
            / f"gridworld_2000ep_200ms_{dataset_seed}seed.npz"
        )
        tensor_Xtemp, tensor_y = extract_dataset_torch(dataset_path)

        run_dir = artifact_root / experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "metrics.json"

        seed_everything(experiment_seed)
        train_generator = create_generator(experiment_seed)

        epoch_history: list[dict[str, object]] = []
        print("\n" + "=" * 80)
        print(f"Starting experiment: {experiment_name}")
        print("=" * 80)

        if input_ch==1:
                tensor_X=to_cnn_1ch(tensor_Xtemp)
        elif input_ch == 3:
                tensor_X=to_cnn_3ch(tensor_Xtemp)
        splits = split_dataset(tensor_X, tensor_y, seed=split_seed)
        train_loader = DataLoader(splits.train,batch_size=batch_size, shuffle=True,generator=train_generator)
        val_loader = DataLoader(splits.val, batch_size=batch_size, shuffle=False,generator=train_generator)
        test_loader = DataLoader(splits.test, batch_size=batch_size, shuffle=False,generator=train_generator)

        model=CNN(
            input_ch=input_ch,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            pooling=pooling,
            dropout=dropout,
            padding=padding,
            fc_hidden=fc_hidden,
                  ).to(device)
      
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

        sync_cuda(device)
        experiment_start=time.perf_counter()
        tensorboard_dir = (
            Path("artifacts")
            / "p4_cnn"
            / "tensorboard"
            / config["name"]
        )
        cumulative_elapsed=0
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_text(
            "Run/config",
            (
                f"dataset_seed={dataset_seed}\n\n"
                f"experiment_seed={experiment_seed}\n\n"
                f"split_seed={split_seed}\n\n"
                f"input_ch={input_ch}\n\n"
                f"conv_channels={conv_channels}\n\n"
                f"fc_hidden={fc_hidden}\n\n"
                f"dropout={dropout}\n\n"
                f"learning_rate={config['learning_rate']}\n\n"
                f"batch_size={batch_size}"
            ),
            0,
        )

        for epoch in range(1,max_epochs+1):
            sync_cuda(device)
            epoch_start=time.perf_counter()
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

            sync_cuda(device)
            epoch_elapsed = time.perf_counter() - epoch_start
            epoch_elapsed = time.perf_counter() - epoch_start
            cumulative_elapsed += epoch_elapsed

            epoch_history.append(
                {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "elapsed_seconds": epoch_elapsed,
                }
            )
            if validation_loss < best_val_loss - min_delta:
                    best_val_loss = validation_loss
                    best_val_accuracy = validation_accuracy
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_time_to_checkpoint = cumulative_elapsed
                    torch.save(
                        {
                            "experiment_name": experiment_name,
                            "model_type": "cnn",
                            "config": config.copy(),
                            "dataset_seed": dataset_seed,
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
                f"epoch {epoch:02d}/{max_epochs} | "
               # f"train loss={train_loss:.4f}, "
               # f"train acc={train_accuracy:.2%} | "
                f"val loss={validation_loss:.4f}, "
                f"val acc={validation_accuracy:.2%}"
            )
            writer.add_scalar(
                "Loss/train",
                train_loss,
                epoch,
            )

            writer.add_scalar(
                "Loss/validation",
                validation_loss,
                epoch,
            )

            writer.add_scalar(
                "Accuracy/train",
                train_accuracy,
                epoch,
            )

            writer.add_scalar(
                "Accuracy/validation",
                validation_accuracy,
                epoch,
            )

            writer.add_scalar(
                "Performance/epoch_seconds",
                epoch_elapsed,
                epoch,
            )  
            writer.add_scalar(
                "Performance/cumulative_seconds",
                cumulative_elapsed,
                epoch,
            )         

        writer.close()
        sync_cuda(device)
        experiment_elapsed = time.perf_counter() - experiment_start

        seconds_per_epoch = experiment_elapsed / len(epoch_history)
        writer.add_scalar(
                        "Performance/time_to_best_seconds",
                        best_time_to_checkpoint,
                        best_epoch,
                    )
        save_json(
                    metrics_path,
                    {
                        "experiment_name": experiment_name,
                        "config": config,
                        "dataset_seed":dataset_seed,
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
                        "elapsed_seconds": experiment_elapsed,
                        "seconds_per_epoch": seconds_per_epoch,
                    },
                )        
        experiment_results.append(
                    {
                        "name": experiment_name,
                        "dataset_seed": dataset_seed,
                        "experiment_seed": experiment_seed,
                        "split_seed": split_seed,
                        "checkpoint_path": checkpoint_path,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_val_loss,
                        "best_validation_accuracy": best_val_accuracy,
                        "elapsed_seconds": experiment_elapsed,
                        "time_to_best_seconds": best_time_to_checkpoint,
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
            f"| time={result['elapsed_seconds']:.1f}s"
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
    print(f"Best model was from model {checkpoint['experiment_name']}")
    print(f"Test loss: {bestLoss:.4f}")
    print(f"Test accuracy: {bestAcc:.2%}")
    

if __name__ == "__main__":
        ensure_final_datasets()
        raise SystemExit(main())