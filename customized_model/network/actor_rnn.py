import torch
import torch.nn as nn

#input current state and action
#output a sequence of future actions

class VehicleActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, rnn_layers, max_action =1):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action

        input_dim = state_dim  # Input: state, command, prev_action
        self.fc_pre = nn.Linear(input_dim,hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.layernorm = nn.LayerNorm(state_dim)

    def forward(self, state, seq_len):
        """
        state: [batch, state_dim]
        command: [batch, cmd_dim]
        rollout_len: int (number of future actions to generate)
        Returns:
            action_seq: [batch, rollout_len, action_dim]
        """
        batch_size = state.size(0)
        device = state.device

        # Initial hidden/cell states
        h = None

        action_seq = []

        x = self.layernorm(state)
        x = self.fc_pre(x)

        for t in range(seq_len):
            out,h = self.lstm(x, h)
            action = self.max_action * torch.tanh(self.fc(out))  # Predict action_t
            action_seq.append(action)

        return torch.cat(action_seq, dim=1)  # [B, T, action_dim]
