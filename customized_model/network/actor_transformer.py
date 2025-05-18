import torch
import torch.nn as nn
from network.utilites import PositionalEncoding
#input current state and action
#output a sequence of future actions

class VehicleActor(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1, max_action=1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_action = max_action
        # Positional encoding
        # self.pos_encoding = nn.Parameter(torch.randn(1, 10000, d_model))  # Max seq length = 1000
        self.pos_encoding = PositionalEncoding(d_model)
        self.layernorm = nn.LayerNorm(state_dim)

        # Input projection: state and action → d_model
        self.state_encoder = nn.Linear(state_dim, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: d_model → state prediction
        self.state_decoder = nn.Linear(d_model, action_dim)

    def forward(self, current_state, seq_len):

        B = current_state.size(0)
        T = seq_len
        current_state = self.layernorm(current_state)
        # Repeat current state for each timestep
        state_token = self.state_encoder(current_state).repeat(1, T, 1)  # [B, T, d_model]
        # Encode actions
        # Combine state and action embeddings
        x = state_token  # [B, T, d_model]
        # x = x + self.pos_encoding[:, :T, :]  # Add positional encoding
        x = self.pos_encoding(x)
        # Pass through transformer
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Decode into future states
        action_seq = self.state_decoder(x)  # [B, T, state_dim]
        action_seq = self.max_action*torch.tanh(action_seq)

        return action_seq