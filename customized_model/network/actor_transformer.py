import torch
import torch.nn as nn
from network.utilites import PositionalEncoding
#input current state and action
#output a sequence of future actions

class VehicleActor(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

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
        """
        current_state: [B, state_dim]
        action_seq:    [B, T, action_dim]
        Returns:
        pred_state_seq: [B, T, state_dim]
        """
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
        action_seq = torch.tanh(action_seq)
        # if torch.isnan(self.state_encoder.weight).any():
        #     print("NaN detected in pred_states!")
        #     exit()
        # with torch.no_grad():
            # print("current_state stats:", ori_current_state.min().item(), ori_current_state.max().item())
            # print("current_state stats:", current_state.min().item(), current_state.max().item())
            # print("encoded state:", state_token.min().item(), state_token.max().item())
            # print("after pos_encoding:", x.min().item(), x.max().item())
            # print("after transformer:", x.min().item(), x.max().item())
            # print("final output before tanh:", action_seq.min().item(), action_seq.max().item())
            # print("has nan:", torch.isnan(action_seq).any().item())
            # print("Any NaNs in state_encoder weights?", torch.isnan(self.state_encoder.weight).any())
            # print("Any Infs in state_encoder weights?", torch.isinf(self.state_encoder.weight).any())
            # print("Any NaNs in bias?", torch.isnan(self.state_encoder.bias).any())

        return action_seq