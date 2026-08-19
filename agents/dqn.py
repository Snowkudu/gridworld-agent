import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from agents.replay_buffer import ReplayBuffer, Transition
from data.representation import to_cnn_3ch
from environment.gridworld import GridWorld as Gridworld
from models.checkpoint import build_model_from_config


@dataclass
class OptimizationMetrics:
    optimization_step: int
    td_loss: float
    predicted_q_mean: float
    bellman_target_mean: float
    target_next_q_mean: float
    target_synced: bool
    parameter_gap: float


class DQNAgent:
    def __init__(
        self,
        config_online: dict[str, Any],
        config_target: dict[str, Any],
        replay_capacity: int,
        batch_size: int,
        gamma: float,
        learning_rate: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        weight_decay: float,
        epsilon_update_interval: int,
        target_sync_interval: int,
        device: torch.device | str = "cpu",
    ):
        self.device = device
        self.online = build_model_from_config(config_online).to(self.device)
        self.target = build_model_from_config(config_target).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.optimization_steps = 0
        self.target_sync_interval = target_sync_interval
        self.epsilon_min = epsilon_min
        self.epsilon_update_interval = epsilon_update_interval
        self.q_transform: Callable[[torch.Tensor], torch.Tensor] | None = None

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: int,
    ):
        s = state.detach().clone()
        n = next_state.detach().clone()
        transition = Transition(
            state=s, action=action, reward=reward, next_state=n, done=done
        )
        tmp_capacity = self.replay_buffer.current_capacity
        self.replay_buffer.push(transition=transition)
        if tmp_capacity != self.replay_buffer.capacity:
            assert self.replay_buffer.current_capacity - 1 == tmp_capacity

    def _build_batch(self, transitions):
        states = torch.stack([i.state for i in transitions])
        next_states = torch.stack([i.next_state for i in transitions])
        states = to_cnn_3ch(states.reshape(len(transitions), -1)).to(self.device)
        next_states = to_cnn_3ch(next_states.reshape(len(transitions), -1)).to(
            self.device
        )
        actions = torch.tensor(
            [t.action for t in transitions],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [t.reward for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [t.done for t in transitions],
            dtype=torch.bool,
            device=self.device,
        )

        return states, actions, rewards, next_states, dones

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._build_batch(transitions)

        q_values = self.online(states)
        actions = actions.unsqueeze(1)
        q_values = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            # next_q_values = self.target(next_states)
            # next_q_values = torch.max(next_q_values, dim=1).values
            not_dones = (~dones).float()
            # target_q_values= rewards+self.gamma+next_q_values*not_dones
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * not_dones

        loss_fn = nn.HuberLoss()
        loss = loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimization_steps += 1
        target_synced = self.optimization_steps % self.target_sync_interval == 0
        if target_synced:
            self.sync_target()

        parameter_gap = 0.0

        for online_param, target_param in zip(
            self.online.parameters(),
            self.target.parameters(),
        ):
            parameter_gap += (
                (online_param.detach() - target_param.detach()).pow(2).sum().item()
            )

        parameter_gap = parameter_gap**0.5

        metrics = OptimizationMetrics(
            optimization_step=self.optimization_steps,
            td_loss=loss.item(),
            predicted_q_mean=q_values.detach().mean().item(),
            bellman_target_mean=target_q_values.mean().item(),
            target_next_q_mean=next_q_values.mean().item(),
            target_synced=target_synced,
            parameter_gap=parameter_gap,
        )

        return loss.item(), metrics

    def update_epsilon(self):
        if self.optimization_steps % self.epsilon_update_interval != 0:
            return
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def select_action(
        self,
        state: Gridworld,
        exploration_fn: Callable[[], int] | None = None,
    ):
        actions = [0, 1, 2, 3]
        if random.random() < self.epsilon:
            if exploration_fn is not None:
                return exploration_fn()

            return random.choice(actions)
            # e random exploration
        else:
            # 1-e
            state_tensor = state.get_state_tensor().reshape(1, -1)

            cnn_state = to_cnn_3ch(state_tensor).to(self.device)

            with torch.no_grad():
                q_values = self.online(cnn_state)
                if self.q_transform is not None:
                    q_values = self.q_transform(q_values)

            action = int(q_values.argmax(dim=1).item())

            return action

    def get_q_values(self, env):
        state = env.get_state_tensor().reshape(1, -1)
        state = to_cnn_3ch(state).to(self.device)

        with torch.no_grad():
            return self.online(state).squeeze(0)


def main() -> int:
    return 0


if __name__ == "__main__":
    main()
