import numpy as np
import matplotlib.pyplot as plt
from actor import ActorTransformer
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
error_states = data[:, [3,5,6,7]]
states = data[:, 15:25] 
actions = data[:, -4:]
error_dim = error_states.shape[1]
state_dim = states.shape[1]
action_dim = actions.shape[1]
# print(error_states.shape)
# print(states.shape)

# plt.plot(actions[:,3], label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
# plt.show()

# print(states.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ActorTransformer(state_dim = state_dim, error_dim = error_dim, 
                         hidden_dim = 32, num_layers = 2,
                         output_dim = action_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss(reduction = 'sum')
# Assuming error_states, states, and actions are already extracted from `data`
training_data = np.hstack((error_states, states, actions))

# rows_with_nan = np.any(np.isnan(training_data), axis=1)
# nan_row_indices = np.where(rows_with_nan)[0]
# print("Rows with NaN:", nan_row_indices)

# exit()
# print(training_data[1,:])
ep_loss = []
seq_len = 20       # sequence length for transformer
batch_size = 64    # number of sequences per batch
num_epochs = 20    # how many passes over the dataset



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
        error_seq = batch[:, :, :error_dim]                             # (B, T, error_dim)
        state_seq = batch[:, :, error_dim:error_dim + state_dim]       # (B, T, state_dim)
        target_actions = batch[:, -1, -action_dim:]                    # (B, action_dim)

        # print(error_seq.shape)
        # print(state_seq.shape)
        current_time_seconds = time.time()
        preds = model(state_seq, error_seq)  # Your model takes (state, error) as inputs

        loss = loss_fn(preds, target_actions)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    ep_loss.append(total_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "actor_transformer.pth")
plt.plot(ep_loss, label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
plt.show()