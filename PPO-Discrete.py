import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
from torch.distributions import Categorical
from collections import deque


def compute_gae(
    next_value: list,
    rewards: list,
    dones: list,
    values: list,
    gamma: float,
    tau: float
) -> List[float]:
    values = values + [next_value]
    gae = 0.0
    returns = deque()
    for i in reversed(range(len(rewards))):
        delta = (
            rewards[i]
            + gamma * values[i + 1] * dones[i] - values[i]
        )
        gae = delta + gamma * tau * dones[i] * gae
        returns.appendleft(gae + values[i])
    return list(returns)


def ppo_iter(
    epoch: int,
    batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
) -> Tuple:
    n = states.size(0)
    for _ in range(epoch):
        for _ in range(n // batch_size):
            idxs = np.random.choice(n, batch_size)
            yield states[idxs, :], actions[idxs], values[idxs], \
                log_probs[idxs], returns[idxs], advantages[idxs]


class Policy_Network(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float
    ) -> None:
        super(Policy_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> Tuple[
            torch.Tensor, torch.distributions.Distribution]:
        dist = Categorical(self.layers(x))
        action = dist.sample()
        return action, dist


class Value_Network(nn.Module):
    def __init__(self, input_size: int, lr: float) -> None:
        super(Value_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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
        tau: float,
        epsilon: float,
        epoch: int,
        rollout_len: int,
        batch_size: int,
        entropy_weight: int
    ) -> None:
        self.policy_network = Policy_Network(input_size, output_size,
                                             policy_lr)
        self.value_network = Value_Network(input_size, value_lr)
        self.input_size = input_size
        self.gamma = gamma
        self.tau = tau
        self.lower = 1 - epsilon
        self.upper = 1 + epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.log_cache = 1 / np.log(output_size)

        self.state_memory = list()
        self.action_memory = list()
        self.reward_memory = list()
        self.value_memory = list()
        self.done_memory = list()
        self.log_prob_memory = list()

    def reset_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.value_memory.clear()
        self.done_memory.clear()
        self.log_prob_memory.clear()

    def choose_action(self, state: np.ndarray, is_train: bool) -> int:
        state = torch.as_tensor(state, dtype=torch.float32)
        action, dist = self.policy_network.forward(state)
        selected_action = action if is_train else dist.mode
        if is_train:
            value = self.value_network.forward(state)
            self.state_memory.append(state)
            self.action_memory.append(selected_action)
            self.value_memory.append(value)
            self.log_prob_memory.append(dist.log_prob(selected_action) *
                                        self.log_cache)
        return selected_action.detach().item()

    def step(self, env: gym.Env, action: np.ndarray) -> Tuple[
            np.ndarray, np.float32, bool]:
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
        self.reward_memory.append(
            torch.from_numpy(np.reshape(reward, (1, -1)).astype(np.float32)))
        self.done_memory.append(
            torch.from_numpy(1 - np.reshape(done, (1, -1)).astype(np.bool8)))
        return next_state, reward, done

    def update(self, next_state: np.ndarray) -> Tuple[float, float]:
        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        next_value = self.value_network.forward(next_state)

        returns = compute_gae(
            next_value, self.reward_memory, self.done_memory,
            self.value_memory, self.gamma, self.tau
        )
        states = torch.cat(self.state_memory).view(-1, self.input_size)
        actions = torch.cat(self.action_memory)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.value_memory).detach()
        log_probs = torch.cat(self.log_prob_memory).detach()
        advantages = returns - values

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for state, action, _, old_log_prob, return_, advantage in \
            ppo_iter(self.epoch, self.batch_size, states, actions, values,
                     log_probs, returns, advantages):
            _, dist = self.policy_network.forward(state)
            log_prob = dist.log_prob(action) * self.log_cache
            ratio = (log_prob - old_log_prob).exp()

            surr_loss = ratio * advantage
            clipped_surr_loss = (
                torch.clamp(ratio, self.lower, self.upper) * advantage
            )
            entropy = dist.entropy().mean()
            policy_loss = -(torch.min(surr_loss, clipped_surr_loss).mean() +
                            self.entropy_weight * entropy)

            value = self.value_network.forward(state)
            value_loss = self.value_network.loss(value, return_)

            self.value_network.optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_network.optimizer.step()

            self.policy_network.optimizer.zero_grad()
            policy_loss.backward()
            self.policy_network.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self.reset_memory()
        return total_policy_loss, total_value_loss

    def train(self, env: gym.Env, iteration: int) -> None:
        i = 0
        while i < iteration:
            state = np.expand_dims(env.reset(), axis=0)
            score = 0.0
            for _ in range(self.rollout_len):
                action = self.choose_action(state, True)
                next_state, reward, done = self.step(env, action)
                state = next_state
                score += reward
                if done:
                    print(f"Iteration: {i + 1}, Score: {score}")
                    state = np.expand_dims(env.reset(), axis=0)
                    score = 0.0
                    i += 1
            policy_loss, value_loss = self.update(next_state)
            print(f"Iteration: {i + 1}, Policy Loss: "
                  f"{policy_loss}, Value Loss: {value_loss}")
        torch.save(self.policy_network.state_dict(), "policy network.pth")
        torch.save(self.value_network.state_dict(), "value network.pth")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        policy_lr=0.001,
        value_lr=0.001,
        gamma=0.99,
        tau=0.85,
        epsilon=0.2,
        epoch=16,
        rollout_len=2048,
        batch_size=32,
        entropy_weight=0.005
    )
    agent.train(env, 10000)
