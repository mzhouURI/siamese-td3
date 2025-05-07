import torch
import torch.nn as nn
import torch.optim as optim

class CriticTransformer(nn.Module):
    def __init__(self, state_dim, error_dim, action_dim, hidden_dim=128, num_heads=4, num_layers=4):
        super(CriticTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings for state, error, and action
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.error_embedding = nn.Linear(error_dim, hidden_dim)
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output layer for Q-value
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, error, action, attention_mask=None):
        # Embed state, error, and action
        # print(f"Action shape: {action.shape}")
        state_emb = self.state_embedding(state)
        error_emb = self.error_embedding(error)

        # print(f"state_emb shape: {state_emb.shape}")
        # print(f"error_emb shape: {error_emb.shape}")
        # print(f"action shape: {action.shape}")
        # Concatenate state, error, and action embeddings (or add them if you prefer)
        x = state_emb + error_emb  # [B, T, H]

        if attention_mask is not None:
            # Convert to key_padding_mask: True = pad
            key_padding_mask = attention_mask == 0  # [B, T]
        else:
            key_padding_mask = None

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, H]

        if attention_mask is not None:
            # Find last valid index per batch
            last_indices = attention_mask.sum(dim=1) - 1  # [B]
        else:
            last_indices = torch.full((x.size(0),), x.size(1)-1, dtype=torch.long, device=x.device)

        last_token = x[torch.arange(x.size(0)), last_indices]  # [B, H]
        # print(f"last_token shape{last_token.shape}")

        # last_token = self.output_layer(x[:, -1, :])     # predict from last token
        # x_cat = last_token + action
        x_cat = torch.cat([last_token, action], dim=-1)  # [B, H + action_dim]
        # print(f"x_cat shape{x_cat.shape}")
        
        q_value = self.q_head(x_cat)  # [B, 1]
        return q_value