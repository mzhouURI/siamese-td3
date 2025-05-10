import torch
import torch.nn as nn
from collections import deque
import random


class RNNActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128, rnn_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(obs_dim, hidden_size, rnn_layers, batch_first=True)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)

    def forward(self, obs_seq, hidden):
        # obs_seq: (B, T, obs_dim)
        rnn_out, hidden = self.rnn(obs_seq, hidden)  # rnn_out: (B, T, hidden_size)
        mean = self.mean_head(rnn_out)
        log_std = self.log_std_head(rnn_out).clamp(-20, 2)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        # log_prob for tanh-transformed normal (approximation)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)  # (B, T, 1)

        return action, log_prob, hidden
    
    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device)
        c = torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device)
        return (h, c)

class RNNCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128, rnn_layers=1):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.rnn = nn.LSTM(input_dim, hidden_size, rnn_layers, batch_first=True)
        self.q_head = nn.Linear(hidden_size, 1)  # Output Q-value at each time step

    def forward(self, obs_seq, action_seq, hidden_state=None):
        # Concatenate inputs along feature dimension
        x = torch.cat([obs_seq, action_seq], dim=-1)  # (B, T, obs_dim + action_dim)
        rnn_out, next_hidden = self.rnn(x, hidden_state)
        q_values = self.q_head(rnn_out)  # (B, T, 1)
        return q_values, next_hidden

    def init_hidden(self, batch_size):
        h = torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device)
        c = torch.zeros(1, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device)
        return (h, c)



class RNNReplayBuffer:
    def __init__(self, buffer_size, seq_len, obs_dim, action_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs_seq, action_seq, reward_seq, next_obs_seq, hidden_state_seq=None):
        """
        Add a sequence of transitions to the buffer.
        All inputs should be torch tensors of shape (seq_len, dim).
        """
        experience = (obs_seq, action_seq, reward_seq, next_obs_seq, hidden_state_seq)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of sequences."""
        batch = random.sample(self.buffer, batch_size)

        obs_seq_batch = []
        action_seq_batch = []
        reward_seq_batch = []
        next_obs_seq_batch = []
        hidden_state_seq_batch = []

        for experience in batch:
            obs_seq, action_seq, reward_seq, next_obs_seq, hidden_state_seq = experience
            obs_seq_batch.append(obs_seq)
            action_seq_batch.append(action_seq)
            reward_seq_batch.append(reward_seq)
            next_obs_seq_batch.append(next_obs_seq)
            hidden_state_seq_batch.append(torch.zeros(1))  # Placeholder if needed

        # Stack and move to device
        obs_seq_batch = torch.stack(obs_seq_batch).to(self.device)             # (B, T, obs_dim)
        action_seq_batch = torch.stack(action_seq_batch).to(self.device)       # (B, T, action_dim)
        reward_seq_batch = torch.stack(reward_seq_batch).to(self.device)       # (B, T, 1)
        next_obs_seq_batch = torch.stack(next_obs_seq_batch).to(self.device)   # (B, T, obs_dim)

        return obs_seq_batch, action_seq_batch, reward_seq_batch, next_obs_seq_batch, None
