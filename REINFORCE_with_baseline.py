from typing import Tuple
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
torch.set_anomaly_enabled(True)


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, lr: float) -> None:
        super(Policy_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Value_Network(nn.Module):
    def __init__(self, input_size: int, lr: float) -> None:
        super(Value_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr)
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        policy_lr: float,
        value_lr: float,
        gamma: float,
    ) -> None:
        self.policy_network = Policy_Network(input_size, output_size,
                                             policy_lr)
        self.value_network = Value_Network(input_size, value_lr)
        self.gamma = gamma
        self.state_memory = list()
        self.log_prob_memory = list()
        self.reward_memory = list()
        self.cache = torch.as_tensor(1 / np.log(output_size),
                                     dtype=torch.float32)

    def choose_action_train(self, state: np.ndarray) -> int:
        self.state_memory.append(state)
        state = torch.as_tensor(state, dtype=torch.float32)
        action_prob_dist = Categorical(self.policy_network.forward(state))
        action = action_prob_dist.sample()
        log_prob = action_prob_dist.log_prob(action) * self.cache
        self.log_prob_memory.append(log_prob)
        return action.detach().item()

    def choose_action_test(self, state: np.ndarray) -> int:
        return self.policy_network.forward(
            torch.as_tensor(state, dtype=torch.float32)
        ).detach().argmax().item()

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self) -> Tuple[float, float]:
        self.policy_network.optimizer.zero_grad()
        self.value_network.optimizer.zero_grad()

        T = len(self.reward_memory)
        returns = torch.zeros(T, dtype=torch.float32)
        returns_sum = 0.0
        for i in range(T - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum

        state_values = self.value_network.forward(
            torch.as_tensor(np.array(self.state_memory), dtype=torch.float32)
        ).view(T)
        value_loss = self.value_network.loss(state_values, returns)
        value_loss.backward()
        self.value_network.optimizer.step()

        policy_loss = torch.tensor(0.0, dtype=torch.float32)
        for return_, state_value, log_prob in zip(returns, state_values,
                                                  self.log_prob_memory):
            policy_loss -= (log_prob * (return_ - state_value).detach())
        policy_loss.backward()
        self.policy_network.optimizer.step()

        self.state_memory.clear()
        self.log_prob_memory.clear()
        self.reward_memory.clear()

        return policy_loss.item(), value_loss.item()

    def train(self, env: gym.Env, iteration: int) -> None:
        for i in range(iteration):
            state = env.reset()
            score = 0.0
            done = False
            while not done:
                action = self.choose_action_train(state)
                state, reward, done, _ = env.step(action)
                self.store_reward(reward)
                score += reward
            policy_loss, value_loss = self.update()
            print(f"Iteration: {i + 1}, Score: {score}, Policy Loss: "
                  f"{policy_loss}, Value Loss: {value_loss}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(
        env.observation_space.shape[0], env.action_space.n, 0.001, 0.001,
        0.99
    )
    agent.train(env, 1000)
