from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from data.dataset import DatasetSplits, extract_dataset_torch, split_dataset
from data.representation import to_cnn_1ch


class CNN(nn.Module):
    def __init__(
        self,
        input_ch: int = 1,
        conv_channels: tuple = (16, 32),
        kernel_size: int = 3,
        pooling: int = 0,
        dropout: float = 0.0,
        padding: int = 1,
        height: int = 10,
        width: int = 10,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        if input_ch <= 0:
            raise ValueError("Input channels must be greater than zero")
        if kernel_size <= 0:
            raise ValueError("Kernel size must be greater than zero")
        if any(ch <= 0 for ch in conv_channels):
            raise ValueError("Convulsions must be greater than zero")
        if pooling not in (0, 1, 2):
            raise ValueError("pooling must be 0, 1, or 2")
        if dropout < 0:
            raise ValueError("dropout cant be negative")
        self.dropout = nn.Dropout(p=dropout)
        self.pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(input_ch, conv_channels[0], kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(
            conv_channels[0], conv_channels[1], kernel_size, padding=padding
        )

        with torch.no_grad():
            dumy = torch.zeros(1, input_ch, height, width)
        features = self._forward_features(dumy)
        flattened_size = features.flatten(1).shape[1]
        self.fc1 = nn.Linear(flattened_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 4)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        # pooling
        if self.pooling >= 1:
            x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        if self.pooling >= 2:
            x = self.pool(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def load_splits(dataset_path: Path, seed: int) -> DatasetSplits:
    """Load the dataset and split it into train, validation, and test sets."""
    X, y = extract_dataset_torch(dataset_path)
    X_cnn = to_cnn_1ch(X)
    dataset_splits = split_dataset(X_cnn, y, seed)
    return dataset_splits


def main() -> int:

    model = CNN()
    example_state = torch.randn(32, 1, 10, 10)

    logits = model(example_state)

    print(model)
    print(f"Input shape:  {example_state.shape}")
    print(f"Output shape: {logits.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
