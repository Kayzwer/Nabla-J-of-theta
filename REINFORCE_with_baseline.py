from typing import Tuple
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float
    ) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_size: int, learning_rate: float) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        policy_network_learning_rate: float,
        value_network_learning_rate: float,
        gamma: float
    ) -> None:
        self.policy_network = PolicyNetwork(input_size, output_size,
                                            policy_network_learning_rate)
        self.value_network = ValueNetwork(input_size,
                                          value_network_learning_rate)
        self.gamma = gamma
        self.state_memory = list()
        self.action_memory = list()
        self.reward_memory = list()

    def reset_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

    def choose_action_train(self, state: np.ndarray) -> int:
        action_probs = torch.distributions.Categorical(
            self.policy_network.forward(
                torch.as_tensor(state, dtype=torch.float32)))
        action = action_probs.sample()
        self.action_memory.append(action_probs.log_prob(action))
        return action.item()

    def choose_action_test(self, state: np.ndarray) -> int:
        return self.policy_network.forward(
            torch.as_tensor(state, dtype=torch.float32)).argmax().item()

    def train(self) -> Tuple[float, float]:
        self.policy_network.optimizer.zero_grad()
        self.value_network.optimizer.zero_grad()

        T = len(self.reward_memory)
        g = np.zeros(T, dtype=np.float32)
        g_sum = 0.0

        for i in range(T - 1, -1, -1):
            g_sum = self.reward_memory[i] + self.gamma * g_sum
            g[i] = g_sum

        b = self.value_network.forward(torch.as_tensor(
            np.array(self.state_memory), dtype=torch.float32)).view(T)
        g = torch.as_tensor(g, dtype=torch.float32)

        with torch.no_grad():
            advantage_values = g - b

        policy_network_loss = 0.0
        for advantage_value, log_prob in zip(advantage_values,
                                             self.action_memory):
            policy_network_loss += -log_prob * advantage_value
        policy_network_loss.backward()
        self.policy_network.optimizer.step()

        value_network_loss = self.value_network.loss(b, g)
        value_network_loss.backward()
        self.value_network.optimizer.step()

        self.reset_memory()

        return policy_network_loss.item(), value_network_loss.item()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.001,
                  0.001, 0.99)

    iteration = 1000
    for i in range(iteration):
        state = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.choose_action_train(state)
            agent.state_memory.append(state)
            state, reward, done, _ = env.step(action)
            agent.reward_memory.append(reward)
            score += reward
        policy_network_loss, value_network_loss = agent.train()
        print(f"Iteration: {i + 1}, Score: {score}, Policy Network Loss: "
              f"{policy_network_loss}, Value Network Loss: "
              f"{value_network_loss}")

    torch.save(agent.policy_network.state_dict(),
               "REINFORCE cartpole policy network.pt")
    torch.save(agent.value_network.state_dict(),
               "REINFORCE cartpole value network.pt")
