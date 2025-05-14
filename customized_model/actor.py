import torch
import torch.nn as nn

#input current state and action
#output a sequence of future actions

class VehicleActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, rnn_layers):
        super().__init__()
        self.action_dim = action_dim
        input_dim = state_dim  # Input: state, command, prev_action
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=rnn_layers, batch_first=True)
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

        for t in range(seq_len):
            # x = torch.cat([state, action], dim=-1)  # [B, D]
            x = state
            x = self.layernorm(x)
            out,h = self.lstm(x, h)
            action = torch.tanh(self.fc(out))  # Predict action_t
            action_seq.append(action)

        return torch.cat(action_seq, dim=1)  # [B, T, action_dim]
