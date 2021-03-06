from typing import Tuple
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(128, output_size)
        self.std_layer = nn.Linear(128, output_size)
        uniform_init(self.mean_layer)
        uniform_init(self.std_layer)
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        x = self.layers(x)
        mean = torch.tanh(self.mean_layer(x)) * 2
        log_std = F.softplus(self.std_layer(x))
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist


class Value_Network(nn.Module):
    def __init__(self, input_size: int, learning_rate: float):
        super(Value_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        uniform_init(self.layers[2])
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

    def choose_action(self, state: np.ndarray, is_train: bool) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32)
        action, dist = self.policy_network.forward(state)
        selected_action = action if is_train else dist.mean
        if is_train:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            entropy = dist.entropy()
            self.transition.extend([state, log_prob, entropy])
        return selected_action.clamp(-2.0, 2.0).detach().numpy()

    def store_info(self, next_state: np.ndarray, reward: float, done: bool
                   ) -> None:
        self.transition.extend([next_state, reward, done])

    def update(self) -> Tuple[float, float]:
        state, log_prob, entropy, next_state, reward, done = self.transition
        self.transition.clear()

        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        q_pred = self.value_network.forward(state)
        q_target = reward + self.gamma * self.value_network.forward(
            next_state) * ~torch.as_tensor(done, dtype=torch.bool)
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
                action = self.choose_action(state, True)
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
    env = gym.make("Pendulum-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.shape[0],
                  1e-4, 1e-3, 0.75, 0.05)
    agent.train(env, 1000)
    # agent.policy_network.load_state_dict(torch.load("policy network.pth"))
    # agent.value_network.load_state_dict(torch.load("value network.pth"))
    # score = 0.0
    # done = False
    # state = env.reset()
    # while not done:
    #     env.render()
    #     action = agent.choose_action(state, False)
    #     state, reward, done, _ = env.step(action)
    #     score += reward
    # print(f"Score: {score}")
