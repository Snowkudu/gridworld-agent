from pathlib import Path

import torch

from agents.dqn import DQNAgent
from models.checkpoint import load_model_from_checkpoint


def transfer_cnn_to_dqn(
    agent: DQNAgent,
    checkpoint_path: str | Path,
    freeze: bool = False,
) -> None:
    pretrained, _ = load_model_from_checkpoint(
        checkpoint_path,
        agent.device,
    )

    with torch.no_grad():
        agent.online.conv1.load_state_dict(pretrained.conv1.state_dict())
        agent.online.conv2.load_state_dict(pretrained.conv2.state_dict())
        agent.online.fc1.load_state_dict(pretrained.fc1.state_dict())

    if freeze:
        for param in agent.online.conv1.parameters():
            param.requires_grad = False

        for param in agent.online.conv2.parameters():
            param.requires_grad = False

        for param in agent.online.fc1.parameters():
            param.requires_grad = False
    agent.sync_target()
