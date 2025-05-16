import torch
import torch.nn as nn
from network.utilites import PositionalEncoding

class VehicleModeler(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Positional encoding
        # self.pos_encoding = nn.Parameter(torch.randn(1, 10000, d_model))  # Max seq length = 1000
        self.pos_encoding = PositionalEncoding(d_model)
        # Input projection: state and action → d_model
        self.layernorm = nn.LayerNorm(state_dim)

        self.state_encoder = nn.Linear(state_dim, d_model)
        self.action_encoder = nn.Linear(action_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: d_model → state prediction
        self.state_decoder = nn.Linear(d_model, state_dim)

    def forward(self, current_state, action_seq):
        """
        current_state: [B, state_dim]
        action_seq:    [B, T, action_dim]
        Returns:
        pred_state_seq: [B, T, state_dim]
        """
        B, T, _ = action_seq.shape
        # Repeat current state for each timestep
        current_state = self.layernorm(current_state)
        state_token = self.state_encoder(current_state).repeat(1, T, 1)  # [B, T, d_model]
        # Encode actions
        action_token = self.action_encoder(action_seq)  # [B, T, d_model]

        # Combine state and action embeddings
        x = state_token + action_token  # [B, T, d_model]
        # x = x + self.pos_encoding[:, :T, :]  # Add positional encoding
        x = self.pos_encoding(x)
        # Pass through transformer
        x = self.transformer_encoder(x)  # [B, T, d_model]

        # Decode into future states
        pred_state_seq = self.state_decoder(x)  # [B, T, state_dim]

        pred_state_seq = self.normalize_sincos(pred_state_seq, 1,2)
        pred_state_seq = self.normalize_sincos(pred_state_seq, 3,4)
        pred_state_seq = self.normalize_sincos(pred_state_seq, 5,6)

        return pred_state_seq

    def normalize_sincos(self, state, cos_idx=1, sin_idx=2):
        sincos = state[:, :, [sin_idx, cos_idx]]
        norm = torch.norm(sincos, dim=-1, keepdim=True) + 1e-8
        sincos_unit = sincos / norm
        state[:, :, sin_idx] = sincos_unit[:, :, 0]
        state[:, :, cos_idx] = sincos_unit[:, :, 1]
        return state
