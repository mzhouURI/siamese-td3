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
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Output layer for Q-value
        self.output_layer = nn.Linear(hidden_dim, 1)  # Single value for Q

    def forward(self, state, error, action):
        # Embed state, error, and action
        # print(f"Action shape: {action.shape}")
        state_emb = self.state_embedding(state)
        error_emb = self.error_embedding(error)
        action_emb = self.action_embedding(action)
        # action_emb = action_emb.unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
        # action_emb = action_emb.expand(-1, state_emb.size(1), -1)  # Shape: (batch_size, seq_len, feature_dim)

        # print(f"state_emb shape: {state_emb.shape}")
        # print(f"error_emb shape: {error_emb.shape}")
        # print(f"action_emb shape: {action_emb.shape}")
        # Concatenate state, error, and action embeddings (or add them if you prefer)
        x = state_emb + error_emb + action_emb  # Alternatively, you can use torch.cat(state_emb, error_emb, action_emb, dim=-1)
        
        # Add a batch dimension (transformer expects (seq_len, batch_size, feature_dim))
        # x = x.unsqueeze(0)  # Adding batch dimension at the beginning
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)
        
        # Get the output (from the first token for simplicity)
        output = self.output_layer(x[:, 0, :])  # Take the output from the first token across the batch
        
        return output
