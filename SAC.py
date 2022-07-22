import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple


def init_layer_uniform(layer: nn.Linear,
                       init_param: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_param, init_param)
    layer.bias.data.uniform_(-init_param, init_param)
    return layer


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.log_std_layer = nn.Linear(128, out_dim)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)
        self.mu_layer = nn.Linear(128, out_dim)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        mu = self.mu_layer(x).tanh()
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, in_dim: int):
        super(QNetwork, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value


class VNetwork(nn.Module):
    def __init__(self, in_dim: int):
        super(VNetwork, self).__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value


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
                 num_random_steps: int, policy_update_freq: int) -> None:
        self.replay_buffer = ReplayBuffer(input_dim, buffer_size, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.tau_dif = 1.0 - tau
        self.output_dim = output_dim
        self.num_random_steps = num_random_steps
        self.policy_update_freq = policy_update_freq

        self.target_entropy = -np.prod((output_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = PolicyNetwork(input_dim, output_dim)
        self.vf = VNetwork(input_dim)
        self.vf_target = VNetwork(input_dim)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf_1 = QNetwork(input_dim + output_dim)
        self.qf_2 = QNetwork(input_dim + output_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=3e-4)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=3e-4)

        self.cur_step = 0

    def choose_action(self, state: np.ndarray, is_train: bool) -> np.ndarray:
        if self.cur_step < self.num_random_steps and is_train:
            selected_action = np.random.uniform(-1.0, 1.0, self.output_dim)
        else:
            selected_action = self.actor.forward(
                torch.as_tensor(state, dtype=torch.float32)
            )[0].detach().numpy()
        self.cur_step += 1
        return selected_action

    def update_target_network(self):
        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_(self.tau * l_param.data + self.tau_dif *
                               t_param.data)

    def update(self) -> Tuple[float, float, float, float]:
        batch = self.replay_buffer.sample()
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        not_dones = ~batch.get("dones")
        new_action, log_prob = self.actor(states)

        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.gamma * v_target * not_dones
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        v_pred = self.vf(states)
        q_pred = torch.min(
            self.qf_1(states, new_action), self.qf_2(states, new_action)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        if self.cur_step % self.policy_update_freq == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target_network()
        else:
            actor_loss = torch.zeros(1)

        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return (actor_loss.item(), qf_loss.item(), vf_loss.item(),
                alpha_loss.item())

    def train(self, env: gym.Env, episode: int, checkpoint: int) -> None:
        for i in range(episode):
            state = env.reset()
            done = False
            score = 0.0
            total_policy_loss, total_q_loss, total_v_loss, total_alpha_loss = \
                0.0, 0.0, 0.0, 0.0
            while not done:
                action = self.choose_action(state, True)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.store_transition(state, action, reward,
                                                    next_state, done)
                score += reward
                state = next_state
                if self.replay_buffer.is_ready() and self.cur_step > \
                   self.num_random_steps:
                    policy_loss, q_loss, v_loss, alpha_loss = self.update()
                    total_policy_loss += policy_loss
                    total_q_loss += q_loss
                    total_v_loss += v_loss
                    total_alpha_loss += alpha_loss
            print(f"Episode: {i + 1}, Score: {score}, Policy Loss: "
                  f"{total_policy_loss}, Q Loss: {total_q_loss}, V Loss: "
                  f"{total_v_loss}, Alpha Loss: {total_alpha_loss}")
            if (i + 1) % checkpoint:
                torch.save(self.actor.state_dict(),
                           "policy network.pt")


if __name__ == "__main__":
    env = ActionNormalizer(gym.make("Pendulum-v1"))
    agent = Agent(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.shape[0],
        buffer_size=100000, batch_size=128, gamma=0.99, tau=5e-3,
        num_random_steps=10000, policy_update_freq=2
    )
    agent.train(env, 1000, 10)
