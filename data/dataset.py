from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import TensorDataset,random_split
from scripts.verify import load_dataset, DatasetValidationError


@dataclass(frozen=True)
class DatasetSplits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset

def extract_dataset_torch(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load X and y as torch tensors without silently changing their dtype or shape."""
    try:
        X, y = load_dataset(path)
    except DatasetValidationError as e:
        raise ValueError(f"Failed to load dataset: {e}") from e

    tensor_X = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y)

    if tensor_X.dtype != torch.float32:
        raise ValueError(f"Expected tensor_X to be float32, but got {tensor_X.dtype}")
    if tensor_y.dtype != torch.int64:
        raise ValueError(f"Expected tensor_y to be int64, but got {tensor_y.dtype}")

    return tensor_X, tensor_y

def split_dataset(
        tensor_X: torch.Tensor,
        tensor_y: torch.Tensor,
        seed: int = 123,
)-> DatasetSplits:

    sample_count = tensor_X.shape[0]
    if sample_count < 10:
        raise ValueError(
            "Dataset must contain at least 10 samples "
            "for an 80/10/10 split."
        )
        #Dataset needs to exceed 10 samples for splitting into train, val, and test sets."

    train_size = int(0.8 * sample_count)
    val_size = int(0.1 * sample_count)
    test_size =sample_count - train_size - val_size  # Ensure all samples are used

    #80% train, 10% validation, 10% test

    complete_dataset = TensorDataset(tensor_X, tensor_y)

    train_dataset, val_dataset, test_dataset = random_split(
        complete_dataset,
        lengths=[train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    #get 3 datasets for training, validation, and testing of length 80%, 10%, and 10% of the original dataset respectively with a random seed of 123 for reproducibility    
    return DatasetSplits(
            train=train_dataset, 
            val=val_dataset, 
            test=test_dataset
         )#return the frozen class DatasetSplits containing the three datasets

def main() -> int:

    path = Path(__file__).parent /  "raw" / "gridworld_2000ep_200ms_123seed.npz"
    print(f"Verifying dataset at path: {path}")
    tensor_X, tensor_y = extract_dataset_torch(path)

    splits=split_dataset(tensor_X, tensor_y, seed=123)
    print(f"Train dataset length: {len(splits.train)}\n")
    print(f"Validation dataset length: {len(splits.val)}\n")
    print(f"Test dataset length: {len(splits.test)}\n")

    print(
        f"DATASET LOADED FROM {path} : \n"
        f"{tensor_X.shape[0]} samples, state shape={tensor_X.shape}, \n"
        f"state dtype={tensor_X.dtype}, action dtype={tensor_y.dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())