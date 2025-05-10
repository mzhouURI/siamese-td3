import torch
import torch.nn as nn


class RNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(RNNCritic, self).__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action, hidden_state=None):
        x = torch.cat([state, action], dim=-1)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        x = self.fc1(lstm_out)
        q_value = self.fc2(x)
        return q_value, hidden_state
