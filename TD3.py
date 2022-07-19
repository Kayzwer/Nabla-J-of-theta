from typing import Dict, Tuple
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, input_dim: int, buffer_size: int, batch_size: int
                 ) -> None:
        self.state_memory = np.zeros((buffer_size, input_dim),
                                     dtype=np.float32)
        self.action_memory = np.zeros(buffer_size, dtype=np.float32)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, input_dim),
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
    def __init__(self, action_dim: int, min_sigma: float = 1.0,
                 max_sigma: float = 1.0, decay_period: int = 1000000) -> None:
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.max_min_dif = max_sigma - min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        sigma = self.max_sigma - (self.max_min_dif) * min(1.0, t /
                                                          self.decay_period)
        return np.random.normal(0, sigma, size=self.action_dim)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 param_init_value: float = 3e-3) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        self.layers[4].weight.data.uniform_(-param_init_value,
                                            param_init_value)
        self.layers[4].bias.data.uniform_(-param_init_value,
                                          param_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, param_init_value: float = 3e-3) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.layers[4].weight.data.uniform_(-param_init_value,
                                            param_init_value)
        self.layers[4].bias.data.uniform_(-param_init_value,
                                          param_init_value)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x_1, x_2), dim=-1)
        return self.layers(x)


class ActionNormalizer(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.scale_factor = (self.action_space.high - self.action_space.low
                             ) / 2.0
        self.reloc_factor = self.action_space.high - self.scale_factor

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action * self.scale_factor + self.reloc_factor,
                       self.action_space.low, self.action_space.high)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip((action - self.reloc_factor) / self.scale_factor,
                       -1.0, 1.0)


class Agent:
    def __init__(self, input_dim: int, output_dim: int, buffer_size: int,
                 batch_size: int, gamma: float, tau: float,
                 num_random_steps: int, policy_update_freq: int,
                 exploration_noise: float, target_policy_noise: float,
                 target_policy_noise_clip: float, policy_lr: float,
                 value_lr: float) -> None:
        self.replay_buffer = ReplayBuffer(input_dim, buffer_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.tau_dif = 1.0 - tau
        self.num_random_steps = num_random_steps
        self.policy_update_freq = policy_update_freq
        self.output_dim = output_dim
        self.exploration_noise = GaussianNoise(output_dim, exploration_noise,
                                               exploration_noise)
        self.target_policy_noise = GaussianNoise(
            output_dim, target_policy_noise, target_policy_noise)
        self.target_policy_noise_clip = target_policy_noise_clip

        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.policy_target_network = PolicyNetwork(input_dim, output_dim)
        self.policy_target_network.load_state_dict(
            self.policy_network.state_dict())
        self.value_network1 = ValueNetwork(input_dim + output_dim)
        self.value_target_network1 = ValueNetwork(input_dim + output_dim)
        self.value_target_network1.load_state_dict(
            self.value_network1.state_dict())
        self.value_network2 = ValueNetwork(input_dim + output_dim)
        self.value_target_network2 = ValueNetwork(input_dim + output_dim)
        self.value_target_network2.load_state_dict(
            self.value_network2.state_dict())
        value_params = list(self.value_network1.parameters()) + \
            list(self.value_network2.parameters())
        self.policy_optimizer = optim.RMSprop(self.policy_network.parameters(),
                                              policy_lr)
        self.value_optimizer = optim.RMSprop(value_params, value_lr)
        self.cur_step = 0

    def choose_action(self, state: np.ndarray, is_train: bool) -> np.ndarray:
        if self.cur_step < self.num_random_steps and is_train:
            selected_action = np.random.uniform(-1.0, 1.0, self.output_dim)
        else:
            selected_action = self.policy_network.forward(
                torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
        if is_train:
            selected_action = np.clip(
                selected_action + self.exploration_noise.sample(), -1.0, 1.0)
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
            self.value_target_network1.parameters(),
            self.value_network1.parameters()
        ):
            target_param.data.copy_(self.tau_dif * target_param.data +
                                    self.tau * cur_param.data)
        for target_param, cur_param in zip(
            self.value_target_network2.parameters(),
            self.value_network2.parameters()
        ):
            target_param.data.copy_(self.tau_dif * target_param.data +
                                    self.tau * cur_param.data)

    def update(self) -> Tuple[float, float]:
        batch = self.replay_buffer.sample()
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        dones = ~batch.get("dones")

        noise = torch.as_tensor(self.target_policy_noise.sample()).clamp(
            -self.target_policy_noise_clip, self.target_policy_noise_clip)
        next_actions = (self.policy_target_network.forward(next_states) +
                        noise).clamp(-1.0, 1.0).type(torch.float32)
        next_values1 = self.value_target_network1.forward(next_states,
                                                          next_actions)
        next_values2 = self.value_target_network2.forward(next_states,
                                                          next_actions)
        next_values = torch.min(next_values1, next_values2)
        cur_returns = (rewards + self.gamma * next_values * dones).detach()

        values1 = self.value_network1.forward(states, actions)
        values2 = self.value_network2.forward(states, actions)
        value1_loss = F.mse_loss(values1, cur_returns)
        value2_loss = F.mse_loss(values2, cur_returns)
        value_loss = value1_loss + value2_loss
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if self.cur_step % self.policy_update_freq == 0:
            policy_loss = -self.value_network1.forward(
                states, self.policy_network.forward(states)).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.update_target_network()
        else:
            policy_loss = torch.zeros(1)
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
                   self.num_random_steps:
                    policy_loss, value_loss = self.update()
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
            print(f"Iteration: {i + 1}, Score: {score}, Policy Loss: "
                  f"{total_policy_loss}, Value Loss: {total_value_loss}")
            if (i + 1) % checkpoint == 0:
                torch.save(self.policy_network.state_dict(),
                           "policy network.pth")
                torch.save(self.value_network1.state_dict(),
                           "value network1.pth")
                torch.save(self.value_network2.state_dict(),
                           "value network2.pth")
                print("Model Saved")


if __name__ == "__main__":
    env = ActionNormalizer(gym.make("Pendulum-v1"))
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0],
                  100000, 128, 0.99, 5e-3, 10000, 2, 0.1, 0.2, 0.5, 3e-4, 1e-3)
    agent.train(env, 1000, 10)
