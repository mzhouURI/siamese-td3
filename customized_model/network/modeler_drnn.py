import torch
import torch.nn as nn


class VehicleModeler(nn.Module):
    def __init__(self, action_dim=4, state_dim=13, hidden_dim=64, rnn_layers=1, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.command_dim = action_dim
        self.state_dim = state_dim

        # Encoder: takes in command sequences
        self.encoder = nn.LSTM(action_dim, hidden_dim, rnn_layers, batch_first=True)

        # Decoder: generates state sequences
        self.decoder_lstm = nn.LSTM(state_dim, hidden_dim, rnn_layers, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, state_dim)
        self.layernorm = nn.LayerNorm(state_dim + action_dim)

    def forward(self, initial_state, commands):
        """
        command_seq: (B, T, 4) – sequence of command inputs
        initial_state: (B, 13) – the initial system state
        returns:
            predicted_states: (B, T, 13) – predicted state sequence
        """
        # batch_size = commands.size(0)
        B, T, _ = commands.shape

        # device = commands.device

        # Encode command sequence
        _, (hidden, cell) = self.encoder(commands)

        # Decode state sequence
        # Start with the initial state (B, 1, 13)
        decoder_input = initial_state
        outputs = []

        for _ in range(T):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))  # out: (B, 1, H)
            pred = self.decoder_fc(out)  # (B, 1, 13)
            pred = self.normalize_sincos(pred, 1,2)
            pred = self.normalize_sincos(pred, 3,4)
            pred = self.normalize_sincos(pred, 5,6)
            outputs.append(pred)
            decoder_input = pred  # use predicted state as input for next step

        predicted_states = torch.cat(outputs, dim=1)  # (B, T, 13)
        return predicted_states
    
    def normalize_sincos(self, state, cos_idx=1, sin_idx=2):
        sincos = state[:, :, [sin_idx, cos_idx]]
        norm = torch.norm(sincos, dim=-1, keepdim=True) + 1e-8
        sincos_unit = sincos / norm
        state[:, :, sin_idx] = sincos_unit[:, :, 0]
        state[:, :, cos_idx] = sincos_unit[:, :, 1]
        return state