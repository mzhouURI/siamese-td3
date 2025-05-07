import torch
import torch.nn as nn

class TransEncoderCritic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=2, seq_len=10):
        super(TransEncoderCritic, self).__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.d_model = d_model

        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        # Input projection (linear layer to d_model)
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to Q-value
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_seq, action_seq):
        """
        Inputs:
            state_seq: (B, T, state_dim)
            action_seq: (B, T, action_dim)
        Returns:
            Q-value: (B, 1)
        """
        x = torch.cat([state_seq, action_seq], dim=-1)  # (B, T, state_dim + action_dim)
        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_encoding.unsqueeze(0)  # (B, T, d_model)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Take the final timestep representation (or use pooling)
        x = x[:, -1, :]  # (B, d_model)

        # Output Q-value
        q_value = self.output_layer(x)  # (B, 1)
        return q_value
