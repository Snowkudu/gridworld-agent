from __future__ import annotations
from pathlib import Path
from pyexpat import model
from typing import Any
from scripts.verify import DatasetValidationError, parse_args, validate_dataset
from data.dataset import extract_dataset_torch, split_dataset, DatasetSplits
import torch
from torch import nn



class MLP(nn.Module):
    def __init__(self,
            input_size: int = 100,
            hidden_sizes: tuple[int, ...] = (128,128),
            num_actions: int = 4,
            dropout: float=0.0,
            ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")

        if num_actions <= 0:
            raise ValueError("num_actions must be greater than zero")

        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("All hidden layer sizes must be greater than zero")
        #redundancy checks

        layers: list[nn.Module] = [
            nn.Flatten(start_dim=1),
        ]
        prev = input_size

        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, hidden),
                    nn.ReLU(),
                ]
            )
            if dropout >0.0:
                layers.append(nn.Dropout(p=dropout))
            prev = hidden
            #appending layers to the model
        layers.append(nn.Linear(prev, num_actions))
        self.network = nn.Sequential(*layers)    

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        #this should return logits for each action given the input states.
        return self.network(states)


def load_splits(dataset_path: Path,seed: int) -> DatasetSplits:
    """Load the dataset and split it into train, validation, and test sets."""
    X, y = extract_dataset_torch(dataset_path)
    dataset_splits = split_dataset(X, y,seed)
    return dataset_splits

def main() -> int:
    
    path = Path(__file__).parent.parent / "data" / "raw" / "gridworld_2000ep_200ms_123seed.npz"
    
    splits=load_splits(path,123)

    model = MLP()
    example_states = torch.zeros(
        size=(32, 100),
        dtype=torch.float32,
    )

    logits = model(example_states)

    print(model)
    print(f"Input shape:  {example_states.shape}")
    print(f"Output shape: {logits.shape}")


    return 0

if __name__ == "__main__":
    raise SystemExit(main())