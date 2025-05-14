import torch
import torch.nn as nn


## input states and actions, output states (RNN learns delta states)
class VehicleModeler(nn.Module):
    def __init__(self, state_dim, command_dim, hidden_dim, rnn_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=state_dim + command_dim,
                           hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        self.layernorm = nn.LayerNorm(state_dim + command_dim)

    def forward(self, initial_state, motor_commands):
        """
        initial_state: (B, state_dim)
        motor_commands: (B, T, command_dim)
        error_state: (B, error_dim)
        """
        B, T, _ = motor_commands.shape
        # state = initial_state.unsqueeze(1)  # (B, 1, state_dim)
        state = initial_state[:,0,:].unsqueeze(1)
        # print(state.shape)
        # exit()
        outputs = []

        h = None  # hidden state, will be initialized as zeros

        for t in range(T):
            cmd_t = motor_commands[:, t:t+1, :]  # (B, 1, command_dim)
            x_t = torch.cat([state, cmd_t], dim=-1)  # (B, 1, input_dim)
            x_t = self.layernorm(x_t)
            out, h = self.rnn(x_t, h)             # (B, 1, hidden_dim)
            delta_state = self.output_layer(out)        # (B, 1, state_dim)
            state = state + delta_state
            outputs.append(state)

        return torch.cat(outputs, dim=1)  # (B, T, state_dim)
