import torch
import torch.nn as nn

# Define the critic head
class CriticHead(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super(CriticHead, self).__init__()
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),  # First linear layer
            nn.ReLU(),                                       # ReLU activation
            nn.Linear(hidden_dim, 1)                          # Output layer (Q-value)
        )

    def forward(self, last_token, action):
        # Concatenate last_token and action along the feature dimension (dim=-1)
        x_cat = torch.cat([last_token, action], dim=-1)  # Shape: [B, hidden_dim + action_dim]
        q_value = self.q_head(x_cat)  # Output: [B, 1] (Q-value for each state-action pair)
        return q_value

# Example tensor shapes
last_token = torch.randn(64, 32)  # [B, hidden_dim]
action = torch.randn(64, 1)       # [B, action_dim]

# Create CriticHead and forward pass
critic = CriticHead(hidden_dim=32, action_dim=1)
q_value = critic(last_token, action)

print(q_value.shape)  # Output: [64, 1]
