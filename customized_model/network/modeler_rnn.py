import torch
import torch.nn as nn


## input states and actions, output states (RNN learns delta states)
class VehicleModeler(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, rnn_layers=1):
        super().__init__()
        self.input_fc = nn.Linear(state_dim + action_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim,hidden_dim)
        self.rnn = nn.LSTM(input_size=state_dim + action_dim,hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, state_dim)
        self.layernorm = nn.LayerNorm(state_dim + action_dim)

    def forward(self, initial_state, commands):

        B, T, _ = commands.shape
        # state = initial_state.unsqueeze(1)  # (B, 1, state_dim)
        # state = initial_state[:,0,:].unsqueeze(1)
        state = initial_state
        # print(state.shape)
        # exit()
        outputs = []

        h = None  # hidden state, will be initialized as zeros

        for t in range(T):
            cmd_t = commands[:, t:t+1, :]  # (B, 1, command_dim)
            x_t = torch.cat([state, cmd_t], dim=-1)  # (B, 1, input_dim)
            x_t = self.layernorm(x_t)
            x_t = self.input_fc(x_t)
            out, h = self.rnn(x_t, h)             # (B, 1, hidden_dim)
            delta_state = self.output_layer(out)        # (B, 1, state_dim)
            state = state + delta_state

            state = self.normalize_sincos(state, 1,2)
            state = self.normalize_sincos(state, 3,4)
            state = self.normalize_sincos(state, 5,6)
           
            outputs.append(state)

        return torch.cat(outputs, dim=1)  # (B, T, state_dim)

    def normalize_sincos(self, state, cos_idx=1, sin_idx=2):
        sincos = state[:, :, [sin_idx, cos_idx]]
        norm = torch.norm(sincos, dim=-1, keepdim=True) + 1e-8
        sincos_unit = sincos / norm
        state[:, :, sin_idx] = sincos_unit[:, :, 0]
        state[:, :, cos_idx] = sincos_unit[:, :, 1]
        return state