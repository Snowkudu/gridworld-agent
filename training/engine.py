from __future__ import annotations

import torch
from torch import inference_mode, nn
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
        states = states.to(device)
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
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model and return the average loss and accuracy.No backprog no gradient updates."""

    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for states, actions in data_loader:
        states = states.to(device)
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


@inference_mode()
def collect_action_diagnostics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    num_actions: int = 4,
) -> dict[str, object]:

    model.eval()

    confusion_matrix = torch.zeros(num_actions, num_actions, dtype=torch.int64)

    total_samples = 0

    for states, actions in data_loader:
        states = states.to(device)
        actions = actions.to(device)

        logits = model(states)
        predictions = logits.argmax(dim=1)

        actions_cpu = actions.cpu()
        predictions_cpu = predictions.cpu()

        flat_pairs = actions_cpu * num_actions + predictions_cpu

        batch_confusion = torch.bincount(
            flat_pairs, minlength=num_actions * num_actions
        ).reshape(num_actions, num_actions)

        confusion_matrix += batch_confusion
        total_samples += actions.shape[0]

    if total_samples == 0:
        raise ValueError("no samples processed during diagnostics")

    true_action_counts = confusion_matrix.sum(dim=1)
    predicted_action_counts = confusion_matrix.sum(dim=0)
    correct_per_action = confusion_matrix.diag()

    per_action_accuracy: list[float | None] = []

    for action in range(num_actions):
        support = int(true_action_counts[action].item())

        if support == 0:
            per_action_accuracy.append(None)
        else:
            accuracy = correct_per_action[action].item() / support
            per_action_accuracy.append(float(accuracy))

    return {
        "sample_count": total_samples,
        "confusion_matrix": confusion_matrix.tolist(),
        "true_action_counts": true_action_counts.tolist(),
        "predicted_action_counts": (predicted_action_counts.tolist()),
        "per_action_accuracy": per_action_accuracy,
    }
