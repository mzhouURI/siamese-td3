import torch
import torch.nn as nn

class VehicleActor(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len, hidden_dim, num_layers=2, max_action=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.max_action = max_action

        input_dim = state_dim
        output_dim = seq_len * action_dim

        self.layernorm = nn.LayerNorm(input_dim)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, state, seq_len):
        x = self.layernorm(state)
        # print(x.shape)
        out = self.net(x)                      # [B, seq_len * action_dim]
        out = out.view(-1, self.seq_len, self.action_dim)  # [B, seq_len, action_dim]
        out = self.max_action*out
        return out
