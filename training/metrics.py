from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset


def save_json(path: str | Path,payload:torch.dist[str, Any]) -> None:
    """Save a dictionary as a JSON file."""
    outputpath = Path(path)
    outputpath.parent.mkdir(parents=True, exist_ok=True)
    temp_path= outputpath.with_suffix(outputpath.suffix+".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            default=str,
        )
    temp_path.replace(outputpath)

def labels_from_subset(
        labels: torch.Tensor, 
        subset: Subset
) -> torch.Tensor:
   indices = torch.as_tensor(subset.indices, dtype=torch.long)
   return labels[indices]

def calculate_majority_baseline(
    labels:torch.Tensor, 
    train_subset : Subset,
    validation_subset : Subset ,
    test_subset: Subset,
    *,
    num_actions : int =4,
)-> dict[str,Any]:
    
    train_labels=labels_from_subset(labels,train_subset)
    validation_labels=labels_from_subset(labels,validation_subset)
    test_labels=labels_from_subset(labels,test_subset)

    train_counts = torch.bincount(
        train_labels,
        minlength=num_actions,
    )
    majority_action=  int(train_counts.argmax().item())

    def constant_accuracy(split_labels: torch.Tensor) -> float:
        if split_labels.numel() == 0:
            raise ValueError("Cannot evaluate an empty label split")

        return float(
            (split_labels == majority_action)
            .float()
            .mean()
            .item()
        )

    return {
        "majority_action": majority_action,
        "training_action_counts": train_counts.tolist(),
        "train_accuracy": constant_accuracy(train_labels),
        "validation_accuracy": constant_accuracy(
            validation_labels
        ),
        "test_accuracy": constant_accuracy(test_labels),
    }
    



    