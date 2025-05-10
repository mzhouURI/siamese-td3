import torch
import torch.nn.functional as F
import copy
from networks import RNNCritic, RNNActor, RNNReplayBuffer


import torch
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, sequence_length=10):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity

    def sample_sequence(self, batch_size):
        # Sample a sequence of transitions from the buffer
        batch = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(self.buffer) - self.sequence_length)
            sequence = self.buffer[start_idx:start_idx + self.sequence_length]
            batch.append(sequence)
        
        # Unzip the batch into individual arrays
        states, actions, rewards, next_states, dones = zip(*[
            [(s, a, r, ns, d) for (s, a, r, ns, d) in sequence] for sequence in batch
        ])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self,
        obs_dim, action_dim, hidden_size=256, rnn_layer = 1, seq_len = 10,
        actor_ckpt = None, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        tau=0.005, gamma=0.99, target_entropy=None, device='cpu'):

        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = RNNActor(obs_dim, action_dim, hidden_size, rnn_layer)

        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")

        self.critic1 = RNNCritic(obs_dim, action_dim, hidden_size, rnn_layer)
        self.critic2 = RNNCritic(obs_dim, action_dim, hidden_size, rnn_layer)
        self.target_critic1 = RNNCritic(obs_dim, action_dim, hidden_size)
        self.target_critic2 = RNNCritic(obs_dim, action_dim, hidden_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        #replay buffer
        self.replay_buffer = RNNReplayBuffer(1000000, seq_len, obs_dim, action_dim, device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy temperature
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

        self.alpha = self.log_alpha.exp().detach()

        # Sync targets
        self._soft_update(self.critic1, self.target_critic1, tau=1.0)
        self._soft_update(self.critic2, self.target_critic2, tau=1.0)

    def select_action(self, obs_seq, hidden_state=None, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            obs_seq = obs_seq.to(self.device).unsqueeze(0)  # Add batch dim
            action, _, _ = self.actor(obs_seq, hidden_state, deterministic)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update(self, batch_size):
        obs_seq, action_seq, reward_seq, next_obs_seq, _ = self.replay_buffer.sample(batch_size)
        B, T, _ = obs_seq.shape

        # Move to device
        obs_seq = obs_seq.to(self.device)
        action_seq = action_seq.to(self.device)
        reward_seq = reward_seq.to(self.device)
        next_obs_seq = next_obs_seq.to(self.device)

        # Init hidden states
        h1 = self.critic1.init_hidden(B)
        h2 = self.critic2.init_hidden(B)
        th1 = self.target_critic1.init_hidden(B)
        th2 = self.target_critic2.init_hidden(B)
        ha = self.actor.init_hidden(B)

        with torch.no_grad():
            next_action, _, next_log_prob = self.actor(next_obs_seq, ha)
            target_q1, _ = self.target_critic1(next_obs_seq, next_action, th1)
            target_q2, _ = self.target_critic2(next_obs_seq, next_action, th2)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward_seq + self.gamma * target_q

        # Critic losses
        q1_pred, _ = self.critic1(obs_seq, action_seq, h1)
        q2_pred, _ = self.critic2(obs_seq, action_seq, h2)

        critic1_loss = F.mse_loss(q1_pred, target_value)
        critic2_loss = F.mse_loss(q2_pred, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        new_action, _, log_prob = self.actor(obs_seq, ha, return_log_prob=True)
        q1_pi, _ = self.critic1(obs_seq, new_action, h1)
        q2_pi, _ = self.critic2(obs_seq, new_action, h2)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Target update
        self._soft_update(self.critic1, self.target_critic1, self.tau)
        self._soft_update(self.critic2, self.target_critic2, self.tau)

        return critic1_loss.item(), critic2_loss.item(),actor_loss.item(), alpha_loss.item(), self.alpha.item()
        

    def _soft_update(self, source, target, tau):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
