from typing import Tuple
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym


def initialize_uniformly(layer: nn.Linear, init_value: float = 3e-3):
    layer.weight.data.uniform_(-init_value, init_value)
    layer.bias.data.uniform_(-init_value, init_value)


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 learning_rate: float) -> None:
        super(Policy_Network, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.mu_layer = nn.Linear(128, output_size)
        self.log_std_layer = nn.Linear(128, output_size)

        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)

        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        x = F.relu(self.fc(x))
        mu = torch.tanh(self.mu_layer(x)) * 2
        log_std = F.softplus(self.log_std_layer(x))
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        return action, dist


class Value_Network(nn.Module):
    def __init__(self, input_size: int, learning_rate: float):
        super(Value_Network, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.out = nn.Linear(128, 1)

        initialize_uniformly(self.out)

        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.fc(state)))


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
            self.transition.extend([state, log_prob])
        return selected_action.clamp(-2.0, 2.0).detach().numpy()

    def update(self) -> Tuple[float, float]:
        state, log_prob, next_state, reward, done = self.transition
        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.bool)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        q_pred = self.value_network(state)
        q_target = reward + self.gamma * self.value_network(next_state) * ~done
        value_loss = F.smooth_l1_loss(q_pred, q_target.detach())

        self.value_network.optimizer.zero_grad()
        value_loss.backward()
        self.value_network.optimizer.step()

        advantage = (q_target - q_pred).detach()
        policy_loss = -advantage * log_prob
        policy_loss += self.entropy_weight * -log_prob

        self.policy_network.optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network.optimizer.step()

        self.transition.clear()

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
                self.transition.extend([next_state, reward, done])
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
                  1e-4, 1e-3, 0.9, 1e-2)
    agent.train(env, 3000)

    # Test agent

    # agent.policy_network.load_state_dict(torch.load("policy network.pth"))
    # agent.value_network.load_state_dict(torch.load("value network.pth"))
    # state = env.reset()
    # score = 0.0
    # done = False
    # while not done:
    #     env.render()
    #     action = agent.choose_action(state, False)
    #     state, reward, done, _ = env.step(action)
    #     score += reward
    # print(score)
    # env.close()
