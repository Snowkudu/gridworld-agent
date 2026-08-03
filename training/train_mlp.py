
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn,save
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.mlp import MLP
from training.engine import train_one_epoch, evaluate
from training.experiment_configs import experiment_configs

from data.dataset import extract_dataset_torch, split_dataset


def main() -> int:
    dataset_path = (
        Path(__file__).parents[1]
        / "data"
        / "raw"
        / "gridworld_2000ep_200ms_123seed.npz"
    )

    tensor_X, tensor_y = extract_dataset_torch(dataset_path)

    splits = split_dataset(tensor_X, tensor_y, seed=123)

    train_loader=DataLoader(splits.train, batch_size=128, shuffle=True)
    val_loader=DataLoader(splits.val, batch_size=128, shuffle=False)
    test_loader=DataLoader(splits.test, batch_size=128, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )#use gpu if available, otherwise use cpu

   # model = MLP().to(device)
    loss_function = nn.CrossEntropyLoss()
  #  optimizer = Adam(model.parameters(), lr=1e-3)
    epochs=50

    checkpoint_dir = Path("data/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_results = []

    for config in experiment_configs:

        experiment_name = config["name"]
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        best_epoch = 0
        checkpoint_path = checkpoint_dir / f"{experiment_name}.pt"

        print("\n" + "=" * 80)
        print(f"Starting experiment: {experiment_name}")
        print(f"Hidden sizes:         {config['hidden_sizes']}")
        print(f"Batch size:           {config['batch_size']}")
        print(f"Learning rate:        {config['learning_rate']}")
        print(f"Weight decay:         {config['weight_decay']}")
        print("=" * 80)
        torch.manual_seed(123)  



        model = MLP(hidden_sizes=config["hidden_sizes"]).to(device)

        optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        train_loader = DataLoader(splits.train, batch_size=config["batch_size"], shuffle=True)

        for epoch in range(1, epochs+1):
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
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_val_accuracy = validation_accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "experiment_name": experiment_name,
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
               

            
            print(
                f"{experiment_name} | "
                f"epoch {epoch:02d}/{epochs} | "
                f"train loss={train_loss:.4f}, "
                f"train acc={train_accuracy:.2%} | "
                f"val loss={validation_loss:.4f}, "
                f"val acc={validation_accuracy:.2%}"
                )
           
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

        
    checkpoint = torch.load(
        best_result["checkpoint_path"],
        map_location=device
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    bestLoss ,bestAcc = evaluate(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
    )
    print(f"Best model was from model {checkpoint['experiment_name']}")
    print(f"Test loss: {bestLoss:.4f}")
    print(f"Test accuracy: {bestAcc:.2%}")

    return 0    

if __name__ == "__main__":
    raise SystemExit(main())



    