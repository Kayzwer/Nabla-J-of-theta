from copy import copy
from random import random
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


class OUNoise:
    def __init__(self, size: int, mean: float, theta: float, std: float
                 ) -> None:
        self.state = np.float64(0.0)
        self.mean = mean * np.ones(size)
        self.theta = theta
        self.std = std
        self.reset()

    def reset(self) -> None:
        self.state = copy(self.mean)

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mean - x) + self.std * np.array(
            [random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 lr: float, param_init_value: float) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
        self.layers[4].weight.data.uniform_(-param_init_value,
                                            param_init_value)
        self.layers[4].bias.data.uniform_(-param_init_value,
                                          param_init_value)
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_size: int, lr: float, param_init_value: float
                 ) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.layers[4].weight.data.uniform_(-param_init_value,
                                            param_init_value)
        self.layers[4].bias.data.uniform_(-param_init_value,
                                          param_init_value)
        self.optimizer = optim.RMSprop(self.parameters(), lr)
        self.loss = nn.MSELoss()

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
    def __init__(self, input_size: int, output_size: int, policy_lr: float,
                 value_lr: float, gamma: float, tau: float, buffer_size: int,
                 batch_size: int, ou_noise_mean: float, ou_noise_theta: float,
                 ou_noise_std: float, num_random_step: int) -> None:
        self.policy_network = PolicyNetwork(input_size, output_size, policy_lr,
                                            3e-3)
        self.policy_target_network = PolicyNetwork(input_size, output_size,
                                                   policy_lr, 3e-3)
        self.value_network = ValueNetwork(input_size + output_size, value_lr,
                                          3e-3)
        self.value_target_network = ValueNetwork(input_size + output_size,
                                                 value_lr, 3e-3)
        self.replay_buffer = ReplayBuffer(input_size, buffer_size, batch_size)
        self.noise = OUNoise(output_size, ou_noise_mean, ou_noise_theta,
                             ou_noise_std)
        self.output_size = output_size
        self.gamma = gamma
        self.tau_dif = 1.0 - tau
        self.tau = tau
        self.num_random_step = num_random_step
        self.cur_step = 0
        self.policy_target_network.load_state_dict(
            self.policy_network.state_dict())
        self.value_target_network.load_state_dict(
            self.value_network.state_dict())

    def choose_action(self, state: np.ndarray, is_train: bool) -> np.ndarray:
        if self.cur_step < self.num_random_step and is_train:
            selected_action = np.random.uniform(-1.0, 1.0, self.output_size)
        else:
            selected_action = self.policy_network(
                torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
        if is_train:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        self.cur_step += 1
        return selected_action

    def update_target_network(self) -> None:
        for target_param, cur_param in zip(
            self.policy_target_network.parameters(),
            self.policy_network.parameters()
        ):
            target_param.data.copy_(self.tau_dif * target_param.data +
                                    self.tau * cur_param.data)
        for target_param, cur_param in zip(
            self.value_target_network.parameters(),
            self.value_network.parameters()
        ):
            target_param.data.copy_(self.tau_dif * target_param.data +
                                    self.tau * cur_param.data)

    def update(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = self.replay_buffer.sample()
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        dones = ~batch.get("dones")

        next_actions = self.policy_target_network.forward(next_states)
        next_values = self.value_target_network.forward(next_states,
                                                        next_actions)
        cur_returns = rewards + self.gamma * next_values * dones

        values = self.value_network.forward(states, actions)
        value_loss = self.value_network.loss(values, cur_returns)
        self.value_network.optimizer.zero_grad()
        value_loss.backward()
        self.value_network.optimizer.step()

        policy_loss = -self.value_network.forward(
            states, self.policy_network(states)).mean()
        self.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network.optimizer.step()

        self.update_target_network()

        return policy_loss.item(), value_loss.item()

    def train(self, env: gym.Env, iteration: int, checkpoint: int) -> None:
        for i in range(iteration):
            state = env.reset()
            done = False
            score = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            while not done:
                action = self.choose_action(state, True)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.store_transition(state, action, reward,
                                                    next_state, done)
                score += reward
                state = next_state
                if self.replay_buffer.is_ready() and self.cur_step > \
                   self.num_random_step:
                    policy_loss, value_loss = self.update()
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
            print(f"Iteration: {i + 1}, Score: {score}, Policy Loss: "
                  f"{total_policy_loss}, Value Loss: {total_value_loss}")
            if (i + 1) % checkpoint == 0:
                torch.save(self.policy_network.state_dict(),
                           "policy network.pth")
                torch.save(self.value_network.state_dict(),
                           "value network.pth")
                print("Model Saved")


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    agent = Agent(
        env.observation_space.shape[0], env.action_space.shape[0], 3e-4, 3e-3,
        0.99, 5e-3, 100000, 128, 0.0, 1.0, 0.1, 10000
    )
    agent.train(env, 1000, 10)
