import torch.nn as nn
import torch

class Critic(nn.Module):
    def __init__(self, state_dim, error_dim, action_dim):
        super().__init__()
        hidden_dim = 128
        input_dim = state_dim + error_dim + action_dim
        self.ln1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, goal, action):
        x = torch.cat([state, goal, action], dim=1)
        # x = self.ln1(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        return self.output_layer(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
