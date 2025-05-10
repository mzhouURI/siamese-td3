import torch
import torch.nn as nn

class RNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(RNNActor, self).__init__()
        self.lstm = nn.LSTM(state_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state, hidden_state=None):
        lstm_out, hidden_state = self.lstm(state, hidden_state)
        x = self.fc1(lstm_out)
        x = self.tanh(x)
        action = self.fc2(x)
        return action, hidden_state
