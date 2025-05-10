import torch
import torch.nn.functional as F
import copy
from critic import RNNCritic
from actor import RNNActor
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
    def __init__(self, state_dim, action_dim, hidden_size=256, alpha=0.2, gamma=0.99, tau=0.005):
        self.actor = RNNActor(state_dim, action_dim, hidden_size)
        self.critic1 = RNNCritic(state_dim, action_dim, hidden_size)
        self.critic2 = RNNCritic(state_dim, action_dim, hidden_size)
        self.target_critic1 = RNNCritic(state_dim, action_dim, hidden_size)
        self.target_critic2 = RNNCritic(state_dim, action_dim, hidden_size)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters())

        self.replay_buffer = ReplayBuffer(1000000, seq_len = 20)

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Update critics
        with torch.no_grad():
            next_action, _ = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = rewards + self.gamma * (1 - dones) * torch.min(target_q1, target_q2)

        q1_value = self.critic1(states, actions)
        q2_value = self.critic2(states, actions)
        critic1_loss = torch.mean((q1_value - target_q)**2)
        critic2_loss = torch.mean((q2_value - target_q)**2)

        # Update critics' weights
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        action, _ = self.actor(states)
        q1_value = self.critic1(states, action)
        q2_value = self.critic2(states, action)
        actor_loss = -torch.mean(torch.min(q1_value, q2_value) - self.alpha * torch.log(action))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target critics
        self.soft_update(self.target_critic1, self.critic1)
        self.soft_update(self.target_critic2, self.critic2)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
