from typing import Tuple
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
torch.set_anomaly_enabled(True)


def uniform_init(layer: nn.Linear, value: float = 3e-3):
    layer.weight.data.uniform_(-value, value)
    layer.bias.data.uniform_(-value, value)


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 learning_rate: float) -> None:
        super(Policy_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        uniform_init(self.layers[4])
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.layers(x)


class Value_Network(nn.Module):
    def __init__(self, input_size: int, learning_rate: float):
        super(Value_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        uniform_init(self.layers[4])
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(self, input_size: int, output_size: int,
                 policy_network_lr: float, value_network_lr: float,
                 gamma: float, entropy_weight: float, n: int) -> None:
        self.policy_network = Policy_Network(input_size, output_size,
                                             policy_network_lr)
        self.value_network = Value_Network(input_size, value_network_lr)
        self.state_memory = list()
        self.log_prob_memory = list()
        self.reward_memory = list()
        self.entropy_memory = list()
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.n = n
        self.log_cache = 1 / np.log(output_size)

    def choose_action(self, state: np.ndarray, is_train: bool) -> int:
        state = torch.as_tensor(state, dtype=torch.float32)
        action_probs = self.policy_network.forward(state)
        action_dist = Categorical(action_probs)
        action = action_dist.sample() if is_train else \
            action_probs.argmax().detach().item()
        if is_train:
            self.state_memory.append(state)
            self.log_prob_memory.append(action_dist.log_prob(action) *
                                        self.log_cache)
            self.entropy_memory.append(action_dist.entropy() * self.log_cache)
        return action.detach().item()

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self, r: torch.Tensor) -> Tuple[float, float]:
        total_value_loss = 0.0
        total_policy_loss = 0.0
        self.value_network.optimizer.zero_grad()
        self.policy_network.optimizer.zero_grad()
        T = len(self.reward_memory)
        for i in range(T - 1, -1, -1):
            r = self.reward_memory[i] + self.gamma * r
            q_pred = self.value_network.forward(self.state_memory[i])
            value_loss = self.value_network.loss(q_pred, r.detach()) / self.n
            total_value_loss += value_loss
            value_loss.backward()

            advantage = (r - q_pred).detach()
            policy_loss = -(advantage * self.log_prob_memory[i] +
                            self.entropy_weight * self.entropy_memory[i]
                            ) / self.n
            total_policy_loss += policy_loss
            policy_loss.backward()
        self.value_network.optimizer.step()
        self.policy_network.optimizer.step()
        self.reset_memory()
        return total_policy_loss.item(), total_value_loss.item()

    def reset_memory(self) -> None:
        self.state_memory.clear()
        self.log_prob_memory.clear()
        self.reward_memory.clear()
        self.entropy_memory.clear()

    def train(self, env: gym.Env, iteration: int) -> None:
        for i in range(iteration):
            state = env.reset()
            score = 0.0
            done = False
            total_p_loss, total_v_loss = 0.0, 0.0
            while not done:
                for _ in range(self.n):
                    action = self.choose_action(state, True)
                    next_state, reward, done, _ = env.step(action)
                    self.store_reward(reward)
                    state = next_state
                    score += reward
                    if done:
                        break
                r = torch.zeros(1, dtype=torch.float32) if done else \
                    self.value_network.forward(torch.as_tensor(
                        state, dtype=torch.float32))
                policy_loss, value_loss = self.update(r)
                total_p_loss += policy_loss
                total_v_loss += value_loss
            print(f"Iteration: {i + 1}, Score: {score}, Policy Loss: "
                  f"{total_p_loss}, Value_Loss: {total_v_loss}")
        torch.save(self.policy_network.state_dict(), "policy network.pth")
        torch.save(self.value_network.state_dict(), "value network.pth")
        print("Model saved")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.001,
                  0.001, 0.75, 0.05, 3)
    agent.train(env, 3000)
