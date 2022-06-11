import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(128, output_size)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float,
        gamma: float
    ) -> None:
        self.network = PolicyNetwork(input_size, output_size, learning_rate)
        self.gamma = gamma
        self.reward_memory = list()
        self.action_memory = list()

    def choose_action_train(self, state: np.ndarray) -> int:
        action_probs = torch.distributions.Categorical(
            F.softmax(self.network.forward(
                torch.as_tensor(state, dtype=torch.float32)), dim=-1))
        action = action_probs.sample()
        self.action_memory.append(action_probs.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def choose_action_test(self, state: np.ndarray) -> int:
        return self.network.forward(
            torch.as_tensor(state, dtype=torch.float32)).argmax().item()

    def train(self) -> float:
        self.network.optimizer.zero_grad()
        g = np.zeros_like(self.reward_memory, dtype=np.float32)
        T = len(g)
        for t in range(T):
            g_sum = 0.0
            n = 0
            for k in range(t, T):
                g_sum += self.reward_memory[k] * self.gamma ** n
                n += 1
            g[t] = g_sum
        g = torch.as_tensor(g, dtype=torch.float32)

        total_loss = 0.0
        loss = 0.0
        for g_i, log_prob in zip(g, self.action_memory):
            loss += -g_i * log_prob
            total_loss += loss
        loss.backward()
        self.network.optimizer.step()

        self.action_memory.clear()
        self.reward_memory.clear()

        return total_loss


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.001,
                  0.99)

    iteration = 100
    for i in range(iteration):
        state = env.reset()
        done = False
        score = 0.0
        while not done:
            action = agent.choose_action_train(state)
            state, reward, done, _ = env.step(action)
            agent.store_reward(reward)
            score += reward
        total_loss = agent.train()
        print(f"Iteration: {i + 1}, Score: {score}, Total Loss: {total_loss}")

    torch.save(agent.network.state_dict(), "REINFORCE cartpole.pt")
