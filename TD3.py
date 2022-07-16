from typing import Dict
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import gym


class ReplayBuffer:
    def __init__(self, input_size: int, buffer_size: int, batch_size: int
                 ) -> None:
        self.state_memory = np.zeros((buffer_size, input_size),
                                     dtype=np.float32)
        self.action_memory = np.zeros(buffer_size, dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, input_size),
                                          dtype=np.float32)
        self.done_memory = np.zeros(buffer_size, dtype=np.bool8)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr, self.cur_size = 0, 0

    def store_transition(self, state: np.ndarray, action: float, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.done_memory[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        selected_indices = np.random.choice(self.cur_size, self.batch_size,
                                            False)
        return {
            "states": torch.as_tensor(self.state_memory[selected_indices],
                                      dtype=torch.float32),
            "actions": torch.as_tensor(
                self.action_memory[selected_indices].reshape(-1, 1),
                dtype=torch.float32),
            "rewards": torch.as_tensor(
                self.reward_memory[selected_indices].reshape(-1, 1),
                dtype=torch.float32),
            "next_states": torch.as_tensor(
                self.next_state_memory[selected_indices], dtype=torch.float32),
            "dones": torch.as_tensor(
                self.done_memory[selected_indices].reshape(-1, 1),
                dtype=torch.bool)
        }

    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class GaussianNoise:
    def __init__(self, action_dim: int, min_std: float, max_std: float,
                 decay_steps: int) -> None:
        self.action_dim = action_dim
        self.max_std = max_std
        self.dif_std = max_std - min_std
        self.decay_steps = decay_steps

    def sample(self, t: int) -> float:
        return np.random.normal(
            0, self.max_std - self.dif_std * min(1.0, t / self.decay_steps),
            self.action_dim)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, init_param: float,
                 lr: float) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.layers[4].weight.data.uniform_(-init_param, init_param)
        self.layers[4].bias.data.uniform_(-init_param, init_param)
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, init_param: float, lr: float) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.layers[4].weight.data.uniform_(-init_param, init_param)
        self.layers[4].bias.data.uniform_(-init_param, init_param)
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x_1, x_2), dim=-1)
        return self.layers(x)


class ActionNormalizer(gym.ActionWrapper):
    def action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


class Agent:
    def __init__(self) -> None:
        pass
