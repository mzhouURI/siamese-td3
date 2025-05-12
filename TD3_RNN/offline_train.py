import numpy as np
import matplotlib.pyplot as plt
from networks import RNNActor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Returns a sequence of length `seq_len`
        sequence = self.data[idx:idx + self.seq_len]
        return torch.tensor(sequence, dtype=torch.float32)

data = np.loadtxt('offline_data/filename1.csv', delimiter=',')
# 0:12: t, ex, ey, ez, e_roll, e_pitch, e_yaw, e_u, e_v, e_w, e_p, e_q, e_r
#13-15: x,y,z,
#16:18: roll, pitch, yaw, 
#19:24 u,v,w,p,q,r
error_states = data[:-1, [3,5,6,7]]
states = data[:-1, 16:25]
# prev_actions = data[:-1,-4:]
states = np.concatenate([states, error_states], axis=1)
# states = data[:, [ 17, 18, 19, 21, 24]] 
actions = data[1:, -4:]

state_dim = states.shape[1]
action_dim = actions.shape[1]
# print(error_states.shape)
# print(states.shape)

# plt.plot(actions[:,3], label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
# plt.show()

# print(states.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = RNNActor(obs_dim = state_dim, action_dim = action_dim,
                 hidden_size = 128, rnn_layers = 2,
                 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = True)
loss_fn = nn.MSELoss(reduction = 'mean')
# Assuming error_states, states, and actions are already extracted from `data`
training_data = np.hstack((states, actions))

# rows_with_nan = np.any(np.isnan(training_data), axis=1)
# nan_row_indices = np.where(rows_with_nan)[0]
# print("Rows with NaN:", nan_row_indices)

# exit()
# print(training_data[1,:])
ep_loss = []
seq_len = 20       # sequence length for transformer
batch_size = 64    # number of sequences per batch
num_epochs = 40    # how many passes over the dataset



# exit()

dataset = SequenceDataset(training_data, seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# print(len(dataloader))
# exit()

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)

        # Separate components
        # states_seq = batch[:, :-1, :state_dim]                             # (B, T, error_dim)
        # exit()
        # actual_action = batch[:, -1, -action_dim:]                    # (B, action_dim)
        # print(batch.shape)
        states_seq = batch [:, : , :state_dim]
        actual_action = batch[:, :, -action_dim:]
        # print(states_seq.shape)
        # print(actual_action.shape)
        rnn_action= model.forward(states_seq)  # Your model takes (state, error) as inputs

        # rnn_action = rnn_action[:, -1, :]  # Shape: (batch_size, state_dim)
        # print(rnn_action)
        # print(actual_action)

        # exit()
        # print(states_seq.shape)

        # exit()
        loss = loss_fn(rnn_action, actual_action)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # print(rnn_action[:,-1,:].detach().cpu().numpy())
    # print(actual_action[:,-1,:].cpu().numpy())

    ep_loss.append(total_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "actor_rnn.pth")
plt.plot(ep_loss, label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
plt.show()