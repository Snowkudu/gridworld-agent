from __future__ import annotations

import torch
from torch import nn,inference_mode
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch and return the average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    
    for states, actions in data_loader:
        states  = states.to(device)
        actions = actions.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(states)

        loss = loss_function(logits, actions)

        loss.backward()
        optimizer.step()

        batch_size = states.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct_predictions += (logits.argmax(dim=1) == actions).sum().item()

        if total_samples == 0:
            raise ValueError("No samples were processed during training.")

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

@inference_mode()
def evaluate(
    model : nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
)-> tuple[float, float]:
    """Evaluate the model and return the average loss and accuracy.No backprog no gradient updates."""
    
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for states, actions in data_loader:
        states  = states.to(device)
        actions = actions.to(device)

        logits = model(states)

        loss = loss_function(logits, actions)

        batch_size = states.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct_predictions += (logits.argmax(dim=1) == actions).sum().item()

    if total_samples == 0:
        raise ValueError("No samples were processed during evaluation.")

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy