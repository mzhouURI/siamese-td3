import torch
import torch.nn as nn
import torch.optim as optim
from position_encoder import PositionalEncoding

class CriticTransformer(nn.Module):
    def __init__(self, state_dim, error_dim, action_dim, hidden_dim=128, num_heads=4, num_layers=4, seq_len=50):
        super(CriticTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Embeddings for state, error, and action
        # self.state_embedding = nn.Linear(state_dim, hidden_dim)
        # self.error_embedding = nn.Linear(error_dim, hidden_dim)
        self.input_proj = nn.Linear(self.state_dim + error_dim + action_dim, hidden_dim)
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output layer for Q-value
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )



    def forward(self, state, error, action, attention_mask=None):

        x = torch.cat([state, error, action], dim=-1)  # (B, T, state_dim + action_dim)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoder(x)  # Add positional encoding

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, T, d_model)

        # Take the final timestep representation (or use pooling)
        x = x[:, -1, :]  # (B, d_model)

        # Output Q-value
        q_value = self.q_head(x)  # (B, 1)

        return q_value