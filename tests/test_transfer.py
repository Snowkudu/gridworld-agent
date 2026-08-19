import torch

from agents.dqn import DQNAgent
from models.checkpoint import build_model_from_config
from training.dqn_transfer import transfer_cnn_to_dqn

CNN_CONFIG = {
    "model_type": "cnn",
    "input_ch": 3,
    "conv_channels": [128, 128],
    "kernel_size": 3,
    "padding": 2,
    "pooling": 1,
    "fc_hidden": 128,
    "dropout": 0.0,
}


def make_agent() -> DQNAgent:
    return DQNAgent(
        config_online=CNN_CONFIG.copy(),
        config_target=CNN_CONFIG.copy(),
        replay_capacity=100,
        batch_size=4,
        gamma=0.90,
        learning_rate=1e-3,
        epsilon_start=1.0,
        epsilon_min=0.10,
        epsilon_decay=0.9999,
        epsilon_update_interval=2,
        weight_decay=0.0,
        target_sync_interval=100,
        device="cpu",
    )


def make_supervised_checkpoint(tmp_path):
    model = build_model_from_config(CNN_CONFIG)

    # Make donor weights unmistakably different from the fresh DQN.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.42)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "config": CNN_CONFIG.copy(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    return checkpoint_path, model


def assert_module_equal(left, right):
    for left_param, right_param in zip(
        left.parameters(),
        right.parameters(),
    ):
        assert torch.equal(left_param, right_param)


def test_transfer_copies_conv_layers(tmp_path):
    agent = make_agent()
    checkpoint_path, pretrained = make_supervised_checkpoint(tmp_path)

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    assert_module_equal(agent.online.conv1, pretrained.conv1)
    assert_module_equal(agent.online.conv2, pretrained.conv2)


def test_transfer_copies_fc1(tmp_path):
    agent = make_agent()
    checkpoint_path, pretrained = make_supervised_checkpoint(tmp_path)

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    assert_module_equal(agent.online.fc1, pretrained.fc1)


def test_transfer_keeps_fresh_q_head(tmp_path):
    agent = make_agent()
    checkpoint_path, pretrained = make_supervised_checkpoint(tmp_path)

    q_weight_before = agent.online.fc2.weight.detach().clone()
    q_bias_before = agent.online.fc2.bias.detach().clone()

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    assert torch.equal(
        agent.online.fc2.weight,
        q_weight_before,
    )
    assert torch.equal(
        agent.online.fc2.bias,
        q_bias_before,
    )

    # Sanity check: donor classifier really was different.
    assert not torch.equal(
        agent.online.fc2.weight,
        pretrained.fc2.weight,
    )


def test_target_synced_after_transfer(tmp_path):
    agent = make_agent()
    checkpoint_path, _ = make_supervised_checkpoint(tmp_path)

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    for online_param, target_param in zip(
        agent.online.parameters(),
        agent.target.parameters(),
    ):
        assert torch.equal(
            online_param,
            target_param,
        )


def test_transfer_preserves_fresh_replay_buffer(tmp_path):
    agent = make_agent()
    checkpoint_path, _ = make_supervised_checkpoint(tmp_path)

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    assert len(agent.replay_buffer) == 0


def test_transfer_preserves_optimizer_parameter_ownership(tmp_path):
    agent = make_agent()
    checkpoint_path, _ = make_supervised_checkpoint(tmp_path)

    online_ids_before = {id(parameter) for parameter in agent.online.parameters()}

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
    )

    online_ids_after = {id(parameter) for parameter in agent.online.parameters()}

    optimizer_ids = {
        id(parameter)
        for group in agent.optimizer.param_groups
        for parameter in group["params"]
    }

    assert online_ids_after == online_ids_before
    assert optimizer_ids == online_ids_after


def test_transfer_freezes_backbone(tmp_path):
    agent = make_agent()
    checkpoint_path, _ = make_supervised_checkpoint(tmp_path)

    transfer_cnn_to_dqn(
        agent,
        checkpoint_path,
        freeze=True,
    )

    for module in (
        agent.online.conv1,
        agent.online.conv2,
        agent.online.fc1,
    ):
        assert all(not parameter.requires_grad for parameter in module.parameters())

    assert all(parameter.requires_grad for parameter in agent.online.fc2.parameters())
