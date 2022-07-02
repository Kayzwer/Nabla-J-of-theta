import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple
from collections import deque, namedtuple
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward',
                               'done', 'value'])
Aux_Memory = namedtuple('Memory', ['state', 'target_value', 'old_values'])


class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))


def create_shuffled_dataloader(data, batch_size) -> DataLoader:
    return DataLoader(ExperienceDataset(data), batch_size=batch_size,
                      shuffle=True)


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, lr: float) -> None:
        super(Policy_Network, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.policy_layers = nn.Sequential(
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )
        self.value_layer = nn.Linear(128, 1)
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.feature_layers(x)
        return self.policy_layers(feature), self.value_layer(feature)


class Value_Network(nn.Module):
    def __init__(self, input_size: int, lr: float) -> None:
        super(Value_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        policy_lr: float,
        value_lr: float,
        rollout_len: int,
        epoch: int,
        aux_epoch: int,
        batch_size: int,
        lamb: float,
        gamma: float,
        entropy_weight: float,
        policy_clip: float,
        value_clip: float
    ) -> None:
        self.policy_network = Policy_Network(input_size, output_size,
                                             policy_lr)
        self.value_network = Value_Network(input_size, value_lr)
        self.rollout_len = rollout_len
        self.epoch = epoch
        self.aux_epoch = aux_epoch
        self.batch_size = batch_size
        self.lamb = lamb
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.policy_clip_upper = 1 + policy_clip
        self.policy_clip_lower = 1 - policy_clip
        self.value_clip = value_clip
        self.log_cache = 1 / np.log(output_size)
        self.memories = deque([])
        self.aux_memories = deque([])

    def choose_action(self, state: torch.Tensor, is_train: bool) -> Tuple[
            int, torch.distributions.Distribution]:
        action_probs, _ = self.policy_network.forward(state)
        dist = Categorical(action_probs)
        return dist.sample() if is_train else dist.mode, dist

    def normalize(self, t: torch.Tensor, eps: float = 1e-5) -> float:
        return (t - t.mean()) / (t.std() + eps)

    def update_network(self, optimizer: torch.optim.Optimizer,
                       loss: torch.Tensor) -> float:
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        return loss.mean()

    def get_value_loss(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
        clip: float
    ) -> torch.Tensor:
        value_clipped = old_values + (values - old_values).clamp(-clip, clip)
        value_loss_1 = (value_clipped.flatten() - rewards) ** 2
        value_loss_2 = (values.flatten() - rewards) ** 2
        return torch.mean(torch.max(value_loss_1, value_loss_2))

    def policy_phase_update(self, next_state: np.ndarray) -> Tuple[float,
                                                                   float]:
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        dones = []
        values = []

        for memory in self.memories:
            states.append(memory.state)
            actions.append(torch.as_tensor(memory.action, dtype=torch.long))
            old_log_probs.append(memory.action_log_prob)
            rewards.append(memory.reward)
            dones.append(1 - memory.done)
            values.append(memory.value)

        next_state = torch.as_tensor(next_state, dtype=torch.float32)
        next_value = self.value_network.forward(next_state).detach()
        values += [next_value]

        returns = deque()
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - \
                values[i]
            gae = delta + self.gamma * self.lamb * dones[i] * gae
            returns.appendleft(gae + values[i])

        states = torch.stack(states).detach()
        actions = torch.stack(actions).detach()
        old_values = torch.stack(values[:-1]).detach()
        old_log_probs = torch.stack(old_log_probs).detach()
        rewards = torch.as_tensor(returns, dtype=torch.float32)

        aux_memory = Aux_Memory(states, rewards, old_values)
        self.aux_memories.append(aux_memory)

        dataloader = create_shuffled_dataloader(
            [states, actions, old_log_probs, rewards, old_values],
            self.batch_size
        )

        total_policy_loss, total_value_loss = 0.0, 0.0
        for _ in range(self.epoch):
            for states, actions, old_log_probs, rewards, old_values in \
                    dataloader:
                action_probs, _ = self.policy_network.forward(states)
                values = self.value_network.forward(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions) * self.log_cache
                entropy = dist.entropy() * self.log_cache

                ratios = (action_log_probs - old_log_probs).exp()
                advantages = self.normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(self.policy_clip_lower,
                                     self.policy_clip_upper) * advantages
                policy_loss = -(torch.min(surr1, surr2) + self.entropy_weight *
                                entropy)
                total_policy_loss += self.update_network(
                    self.policy_network.optimizer, policy_loss)

                value_loss = self.get_value_loss(values, rewards, old_values,
                                                 self.value_clip)
                total_value_loss += self.update_network(
                    self.value_network.optimizer, value_loss)
        self.memories.clear()
        return total_policy_loss.item(), total_value_loss.item()

    def value_phase_update(self) -> Tuple[float, float]:
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in self.aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        old_action_probs, _ = self.policy_network.forward(states)
        old_action_probs.detach_()
        dataloader = create_shuffled_dataloader(
            [states, old_action_probs, rewards, old_values], self.batch_size)
        total_policy_loss, total_value_loss = 0.0, 0.0
        for _ in range(self.aux_epoch):
            for states, old_action_probs, rewards, old_values in dataloader:
                action_probs, policy_values = self.policy_network.forward(
                    states)
                action_log_probs = action_probs.log() * self.log_cache

                aux_loss = self.get_value_loss(policy_values, rewards,
                                               old_values, self.value_clip)
                kl_loss = F.kl_div(action_log_probs, old_action_probs,
                                   reduction='batchmean')
                policy_loss = aux_loss + kl_loss
                total_policy_loss += self.update_network(
                    self.policy_network.optimizer, policy_loss)

                values = self.value_network.forward(states)
                value_loss = self.get_value_loss(values, rewards, old_values,
                                                 self.value_clip)
                total_value_loss += self.update_network(
                    self.value_network.optimizer, value_loss)
        self.aux_memories.clear()
        return total_policy_loss.item(), total_value_loss.item()

    def train(self, env: gym.Env, iteration: int) -> None:
        i = 0
        while i < iteration:
            state = env.reset()
            score = 0.0
            for _ in range(self.rollout_len):
                state = torch.as_tensor(state, dtype=torch.float32)
                action, dist = self.choose_action(state, True)
                action_log_prob = dist.log_prob(action) * self.log_cache
                value = self.value_network.forward(state)
                action = action.item()
                next_state, reward, done, _ = env.step(action)
                memory = Memory(state, action, action_log_prob, reward,
                                done, value)
                self.memories.append(memory)
                score += reward
                state = next_state

                if done:
                    print(f"Iteration: {i + 1}, Score: {score}")
                    i + 1
                    env.reset()
                    score = 0.0
            ph_policy_loss, ph_value_loss = self.policy_phase_update(
                next_state)
            vh_policy_loss, vh_value_loss = self.value_phase_update()
            print(f"Policy Phase --- Iteration: {i + 1}, Policy Loss:"
                  f" {ph_policy_loss}, Value Loss: {ph_value_loss}")
            print(f"Value Phase --- Iteration: {i + 1}, Policy Loss: "
                  f"{vh_policy_loss}, Value Loss: {vh_value_loss}")
            if (i + 1) % 50 == 0:
                torch.save(self.policy_network.state_dict(),
                           "policy network.pth")
                torch.save(self.value_network.state_dict(),
                           "value network.pth")
                print("Model Saved")


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = Agent(
        env.observation_space.shape[0], env.action_space.n, 0.0005, 0.0005,
        500, 1, 6, 16, 0.95, 0.99, 0.01, 0.2, 0.4
    )
    agent.train(env, 1000)
