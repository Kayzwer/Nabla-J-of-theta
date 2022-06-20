from typing import Tuple
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        uniform_init(self.layers[6])
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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        uniform_init(self.layers[6])
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(self, input_size: int, output_size: int,
                 policy_network_lr: float, value_network_lr: float,
                 gamma: float, entropy_weight: float) -> None:
        self.policy_network = Policy_Network(input_size, output_size,
                                             policy_network_lr)
        self.value_network = Value_Network(input_size, value_network_lr)
        self.transition = list()
        self.gamma = gamma
        self.entropy_weight = entropy_weight

    def choose_action(self, state: np.ndarray) -> int:
        state = torch.as_tensor(state, dtype=torch.float32)
        action_probs = Categorical(self.policy_network.forward(state))
        action = action_probs.sample()
        entropy = action_probs.entropy()
        self.transition.extend([state, action_probs.log_prob(action), entropy])
        return action.detach().item()

    def choose_action_test(self, state: np.ndarray) -> int:
        return self.policy_network.forward(
            torch.as_tensor(state, dtype=torch.float32)
        ).argmax().detach().item()

    def store_info(self, next_state: np.ndarray, reward: float, done: bool
                   ) -> None:
        self.transition.extend([next_state, reward, done])

    def update(self) -> Tuple[float, float]:
        state, log_prob, entropy, next_state, reward, done = self.transition
        self.transition.clear()

        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        q_pred = self.value_network.forward(state)
        q_target = reward + self.gamma * self.value_network(next_state) * ~done
        value_loss = self.value_network.loss(q_pred, q_target.detach())

        self.value_network.optimizer.zero_grad()
        value_loss.backward()
        self.value_network.optimizer.step()

        advantage = (q_target - q_pred).detach()
        policy_loss = -(advantage * log_prob + self.entropy_weight * entropy)

        self.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self, env: gym.Env, iteration: int) -> None:
        for i in range(iteration):
            state = env.reset()
            total_p_loss = 0.0
            total_v_loss = 0.0
            score = 0.0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.store_info(next_state, reward, done)
                policy_loss, value_loss = self.update()
                total_p_loss += policy_loss
                total_v_loss += value_loss
                score += reward
                state = next_state
            print(f"Iteration: {i + 1}, Score: {score}, Policy Loss: "
                  f"{total_p_loss}, Value Loss: {total_v_loss}")
        torch.save(self.policy_network.state_dict(), "policy network.pth")
        torch.save(self.value_network.state_dict(), "value network.pth")
        print("Model saved")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(env.observation_space.shape[0], env.action_space.n,
                  1e-4, 1e-4, 0.99, 1e-2)
    agent.train(env, 3000)
