import torch
from torch import nn

from agents.dqn import DQNAgent

DQN_CNN_DEFAULTS = {
    "model_type": "cnn",
    "input_ch": 3,
    "conv_channels": [16, 32],
    "kernel_size": 3,
    "padding": 1,
    "pooling": 0,
    "fc_hidden": 128,
    "dropout": 0.0,
}


def make_agent(
    epsilon_start: float = 1.0,
) -> DQNAgent:
    return DQNAgent(
        config_online=DQN_CNN_DEFAULTS.copy(),
        config_target=DQN_CNN_DEFAULTS.copy(),
        replay_capacity=100,
        batch_size=128,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon_start=epsilon_start,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        weight_decay=0.0,
        epsilon_update_interval=1,
        target_sync_interval=100,
        device="cpu",
    )


class DummyWorld:
    def __init__(self):
        self.grid = torch.zeros((10, 10), dtype=torch.float32)
        self.grid[0, 0] = 1  # agent
        self.grid[9, 9] = 2  # goal
        self.grid[0, 1] = -1  # obstacle

    def get_state_tensor(self):
        return self.grid.clone()


class FixedQNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()

        # Parameter keeps this a real nn.Module with parameters.
        self.q_values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.q_values.unsqueeze(0).expand(x.shape[0], -1)


def test_online_and_target_start_equal():
    agent = make_agent()

    for online_param, target_param in zip(
        agent.online.parameters(),
        agent.target.parameters(),
    ):
        assert torch.equal(online_param, target_param)


def test_optimizer_only_contains_online_parameters():
    agent = make_agent()

    online_ids = {id(param) for param in agent.online.parameters()}

    target_ids = {id(param) for param in agent.target.parameters()}

    optimizer_ids = {
        id(param) for group in agent.optimizer.param_groups for param in group["params"]
    }

    assert optimizer_ids == online_ids
    assert optimizer_ids.isdisjoint(target_ids)


def test_epsilon_one_returns_action_in_full_action_space():
    agent = make_agent(epsilon_start=1.0)
    world = DummyWorld()

    for _ in range(100):
        action = agent.select_action(world)

        assert action in (0, 1, 2, 3)


def test_exploration_uses_full_action_space():
    agent = make_agent()
    agent.epsilon = 1.0
    world = DummyWorld()

    actions = {agent.select_action(world) for _ in range(500)}

    assert actions == {0, 1, 2, 3}


def test_epsilon_zero_uses_highest_q_value():
    agent = make_agent(epsilon_start=0.0)

    agent.online = FixedQNetwork([1.0, -3.0, 8.0, 2.0])

    world = DummyWorld()

    action = agent.select_action(world)

    assert action == 2


def test_greedy_selection_does_not_mask_actions():
    agent = make_agent(epsilon_start=0.0)

    # Action 1 could correspond to a collision in a real world.
    # DQN is still allowed to choose it.
    agent.online = FixedQNetwork([1.0, 100.0, 8.0, 2.0])

    world = DummyWorld()

    action = agent.select_action(world)

    assert action == 1


def test_select_action_does_not_write_replay():
    agent = make_agent(epsilon_start=1.0)
    world = DummyWorld()

    size_before = len(agent.replay_buffer)

    agent.select_action(world)

    assert len(agent.replay_buffer) == size_before


def test_three_channel_cnn_accepts_world_state():
    agent = make_agent(epsilon_start=0.0)
    world = DummyWorld()

    action = agent.select_action(world)

    assert action in (0, 1, 2, 3)


from agents.replay_buffer import Transition


def test_store_transition_increases_replay_size():
    agent = make_agent()

    state = torch.zeros((10, 10))
    next_state = torch.ones((10, 10))

    agent.store_transition(
        state=state,
        action=2,
        reward=-1.0,
        next_state=next_state,
        done=False,
    )

    assert len(agent.replay_buffer) == 1


def test_store_transition_keeps_state_snapshot():
    agent = make_agent()

    state = torch.zeros((10, 10))
    next_state = torch.ones((10, 10))

    agent.store_transition(
        state=state,
        action=1,
        reward=-1.0,
        next_state=next_state,
        done=False,
    )

    # Mutate originals AFTER storage.
    state[0, 0] = 99
    next_state[0, 0] = 99

    stored = agent.replay_buffer.buffer[0]

    assert stored.state[0, 0].item() != 99
    assert stored.next_state[0, 0].item() != 99


def test_build_batch_has_expected_shapes():
    agent = make_agent()

    transitions = [
        Transition(
            state=torch.zeros((10, 10)),
            action=0,
            reward=-1.0,
            next_state=torch.ones((10, 10)),
            done=False,
        ),
        Transition(
            state=torch.ones((10, 10)),
            action=3,
            reward=10.0,
            next_state=torch.zeros((10, 10)),
            done=True,
        ),
    ]

    states, actions, rewards, next_states, dones = agent._build_batch(transitions)

    assert states.shape == (2, 3, 10, 10)
    assert actions.shape == (2,)
    assert rewards.shape == (2,)
    assert next_states.shape == (2, 3, 10, 10)
    assert dones.shape == (2,)


def test_build_batch_preserves_transition_values():
    agent = make_agent()

    transitions = [
        Transition(
            state=torch.zeros((10, 10)),
            action=1,
            reward=-2.5,
            next_state=torch.ones((10, 10)),
            done=False,
        ),
        Transition(
            state=torch.ones((10, 10)),
            action=3,
            reward=10.0,
            next_state=torch.zeros((10, 10)),
            done=True,
        ),
    ]

    _, actions, rewards, _, dones = agent._build_batch(transitions)

    assert actions.tolist() == [1, 3]
    assert rewards.tolist() == [-2.5, 10.0]
    assert dones.tolist() == [False, True]


def test_optimize_changes_online_but_not_target():
    agent = make_agent()
    agent.batch_size = 2

    agent.store_transition(
        state=torch.zeros((10, 10)),
        action=0,
        reward=-1.0,
        next_state=torch.ones((10, 10)),
        done=False,
    )

    agent.store_transition(
        state=torch.ones((10, 10)),
        action=1,
        reward=10.0,
        next_state=torch.zeros((10, 10)),
        done=True,
    )

    online_before = [param.detach().clone() for param in agent.online.parameters()]

    target_before = [param.detach().clone() for param in agent.target.parameters()]

    result = agent.optimize_model()

    assert result is not None

    loss, metrics = result

    online_after = list(agent.online.parameters())
    target_after = list(agent.target.parameters())

    assert loss >= 0.0
    assert metrics.optimization_step == 1

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            online_before,
            online_after,
        )
    )

    assert all(
        torch.equal(before, after)
        for before, after in zip(
            target_before,
            target_after,
        )
    )


def test_optimize_does_nothing_before_replay_warmup():
    agent = make_agent()
    agent.batch_size = 4

    agent.store_transition(
        state=torch.zeros((10, 10)),
        action=2,
        reward=-1.0,
        next_state=torch.ones((10, 10)),
        done=False,
    )

    online_before = [param.detach().clone() for param in agent.online.parameters()]

    result = agent.optimize_model()

    online_after = list(agent.online.parameters())

    assert result is None

    assert all(
        torch.equal(before, after) for before, after in zip(online_before, online_after)
    )


def test_sync_target_copies_online_weights():
    agent = make_agent()

    # Force online and target apart.
    with torch.no_grad():
        for param in agent.online.parameters():
            param.add_(1.0)

    assert any(
        not torch.equal(online_param, target_param)
        for online_param, target_param in zip(
            agent.online.parameters(),
            agent.target.parameters(),
        )
    )

    agent.sync_target()

    assert all(
        torch.equal(online_param, target_param)
        for online_param, target_param in zip(
            agent.online.parameters(),
            agent.target.parameters(),
        )
    )


def test_optimization_step_counter_only_increments_on_real_update():
    agent = make_agent()
    agent.batch_size = 2

    assert agent.optimization_steps == 0

    # Not enough replay yet.
    agent.store_transition(
        state=torch.zeros((10, 10)),
        action=0,
        reward=-1.0,
        next_state=torch.ones((10, 10)),
        done=False,
    )

    agent.optimize_model()

    assert agent.optimization_steps == 0

    # Now enough replay for a real update.
    agent.store_transition(
        state=torch.ones((10, 10)),
        action=1,
        reward=10.0,
        next_state=torch.zeros((10, 10)),
        done=True,
    )

    agent.optimize_model()

    assert agent.optimization_steps == 1


def test_target_sync_happens_at_configured_optimization_interval():
    agent = make_agent()
    agent.batch_size = 2
    agent.target_sync_interval = 2

    for i in range(2):
        agent.store_transition(
            state=torch.full((10, 10), float(i)),
            action=i,
            reward=-1.0,
            next_state=torch.full((10, 10), float(i + 1)),
            done=False,
        )

    # First optimization: online changes, target should remain stale.
    agent.optimize_model()

    assert agent.optimization_steps == 1

    assert any(
        not torch.equal(online_param, target_param)
        for online_param, target_param in zip(
            agent.online.parameters(),
            agent.target.parameters(),
        )
    )

    # Second optimization should trigger sync.
    agent.optimize_model()

    assert agent.optimization_steps == 2

    assert all(
        torch.equal(online_param, target_param)
        for online_param, target_param in zip(
            agent.online.parameters(),
            agent.target.parameters(),
        )
    )


def test_update_epsilon_decays_epsilon():
    agent = make_agent()

    agent.epsilon = 1.0
    agent.epsilon_decay = 0.5
    agent.epsilon_min = 0.1

    agent.update_epsilon()

    assert agent.epsilon == 0.5


def test_update_epsilon_never_goes_below_minimum():
    agent = make_agent()

    agent.epsilon = 0.15
    agent.epsilon_decay = 0.5
    agent.epsilon_min = 0.1

    agent.update_epsilon()

    assert agent.epsilon == 0.1

    agent.update_epsilon()

    assert agent.epsilon == 0.1
