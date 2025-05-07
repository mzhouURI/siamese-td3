import torch
import torch.nn as nn
import torch.optim as optim

# Define the Transformer model
class ActorTransformer(nn.Module):
    def __init__(self, state_dim, error_dim, output_dim=4, hidden_dim=128, num_heads=4, num_layers=4):
        super(ActorTransformer, self).__init__()
        
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.output_dim = output_dim
        
        # Input embeddings (for state and error)
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.error_embedding = nn.Linear(error_dim, hidden_dim)
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers,)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, state, error):
        # Embed the state and error
        state_emb = self.state_embedding(state)
        error_emb = self.error_embedding(error)
        # print(state_emb.shape)
        # print(error_emb.shape)
        # Concatenate the state and error embeddings (if needed for your task)
        x = state_emb + error_emb  # you could use torch.cat(state_emb, error_emb, dim=-1) instead if required
        # print(x.shape)
        # Add a batch dimension (transformer expects (seq_len, batch_size, feature_dim))
        # x = x.unsqueeze(0)  # Adding batch dimension at the beginning
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Get the output (from the first token for simplicity)
        output = self.output_layer(x[:, -1, :])     # predict from last token
        return output
    