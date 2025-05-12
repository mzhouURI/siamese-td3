import torch
import torch.nn as nn
from collections import deque
import random
import torch.nn.functional as F

class RNNActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128, rnn_layers=1):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, hidden_size, rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_dim)

    def forward(self, obs_seq, hidden = None):
        # obs_seq: (B, T, obs_dim)
        self.rnn.flatten_parameters()
        rnn_out, hidden = self.rnn(obs_seq, hidden)  # rnn_out: (B, T, hidden_size)

        # rnn_out = rnn_out[:,-1,:]  #grab last one?

        action = self.fc(rnn_out)  # (B, T, action_dim)

        # action = last_out[:, -1, :].clone()  # Ensures it's not a view

        return action, hidden
    

    def init_hidden(self, batch_size):
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        device = next(self.parameters()).device

        h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # c = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        return h

class RNNCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=128, rnn_layers=1):
        super().__init__()
        input_dim = obs_dim
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(input_dim, hidden_size, rnn_layers, batch_first=True)
        self.q_head = nn.Linear(hidden_size, 1)  # Single Q-value output

    
    def forward(self, obs_seq, hidden=None):
        # Concatenate obs and action along feature dim: (B, T, obs+action)
        # x = torch.cat([obs_seq, action_seq], dim=-1)
        self.rnn.flatten_parameters()
        rnn_out, hidden = self.rnn(obs_seq, hidden)  # rnn_out: (B, T, hidden_size)

        # rnn_out = rnn_out[:,-1,:]  #grab last one?
        
        q_values = self.q_head(rnn_out)  # (B, T, 1)
        # q = q_values[:, -1, :].clone()  # Ensures it's not a view

        return q_values, hidden

    def init_hidden(self, batch_size):
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        device = next(self.parameters()).device
        h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        # c = torch.zeros(num_layers, batch_size, hidden_size).to(device)
        return h

class RNNReplayBuffer:
    def __init__(self, buffer_size, seq_len, obs_dim, action_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs_seq, action_seq, reward_seq, next_obs_seq, actor_hidden):
        """
        Add a sequence of transitions to the buffer.
        All inputs should be torch tensors of shape (seq_len, dim).
        """
        # if any(len(seq) == 0 for seq in (obs_seq, action_seq, reward_seq, next_obs_seq)):
        #     print("Warning: One or more input sequences are empty. Skipping.")
        #     exit()
        # print(action_seq)
        # exit()
        # return  # or raise an exception if preferred
        experience = (obs_seq, action_seq, reward_seq, next_obs_seq, actor_hidden)
 
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
            obs_seq, action_seq, reward_seq, next_obs_seq, actor_hidden = experience
            
            obs_seq = torch.stack(list(obs_seq))  # Ensure proper dtype            
            action_seq = torch.stack(list(action_seq))
            reward_seq = torch.stack(list(reward_seq))
            next_obs_seq = torch.stack(list(next_obs_seq))

            # print(actor_hidden[0])
            tensor_list = [entry[0] for entry in actor_hidden]  # Extract tensors from tuples
            actor_hidden = torch.stack(tensor_list).to(self.device)
            # print(actor_hidden.shape)
            obs_seq_batch.append(obs_seq)
            action_seq_batch.append(action_seq)
            reward_seq_batch.append(reward_seq)
            next_obs_seq_batch.append(next_obs_seq)
            hidden_state_seq_batch.append(actor_hidden)  # Placeholder if needed


        # Stack and move to device
        obs_seq_batch = torch.stack(obs_seq_batch).to(self.device)             # (B, T, obs_dim)
        action_seq_batch = torch.stack(action_seq_batch).to(self.device)       # (B, T, action_dim)
        reward_seq_batch = torch.stack(reward_seq_batch).to(self.device)       # (B, T, 1)
        next_obs_seq_batch = torch.stack(next_obs_seq_batch).to(self.device)   # (B, T, obs_dim)
        # print(hidden_state_seq_batch.shape)
        hidden_state_seq_batch = torch.stack(hidden_state_seq_batch, dim = 0).to(self.device) 
        # print(hidden_state_seq_batch.shape)

        # hidden_state_seq_batch = hidden_state_seq_batch.unsqueeze(2)
        hidden_state_seq_batch = hidden_state_seq_batch.permute(1,0,2)
        hidden_state_seq_batch = hidden_state_seq_batch.contiguous()
        # print(hidden_state_seq_batch.shape)

        # hidden_state_seq_batch = torch.stack(hidden_state_seq_batch).to(self.device)   # (B, T, obs_dim)


        return obs_seq_batch, action_seq_batch, reward_seq_batch, next_obs_seq_batch, hidden_state_seq_batch
        
