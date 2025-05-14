import numpy as np
import matplotlib.pyplot as plt
from modeler import VehicleModeler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
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
# states = data[:-1, 15:25]
states = data[:-1, [15,17,18,19,23,24]]

actions = data[:-1, -4:]
# new_states = data[1:, 15:25]
new_states = data[:-1, [15,17,18,19,23,24]]


state_dim = states.shape[1]
command_dim = actions.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VehicleModeler(state_dim = state_dim, command_dim = command_dim,
                 hidden_dim = 128, rnn_layers = 2,
                 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = True)
loss_fn = nn.MSELoss(reduction = 'mean')
# Assuming error_states, states, and actions are already extracted from `data`
training_data = np.hstack((states, new_states, actions))


ep_loss = []
seq_len = 20       # sequence length for transformer
batch_size = 128    # number of sequences per batch
num_epochs = 40    # how many passes over the dataset


dataset = SequenceDataset(training_data, seq_len=seq_len)

# Split sizes (e.g., 80% train, 20% validation)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

# Randomly split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# print(len(dataloader))
# exit()
# plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

ep_train_loss = []
ep_val_loss = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)


        states_seq = batch [:, : , :state_dim]
        action_seq = batch[:, :, -command_dim:]
        new_state_seq = batch[:,:,state_dim:2*state_dim]

        pred_new_state_seq= model.forward(states_seq, action_seq)  # Your model takes (state, error) as inputs
        # print(pred_new_state_seq.shape)
        # exit()
        loss = loss_fn(new_state_seq, pred_new_state_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    # --- VALIDATION ---
    model.eval()
    batch_count = 1
    print(epoch)
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            states_seq = batch[:, :, :state_dim]
            action_seq = batch[:, :, -command_dim:]
            new_state_seq = batch[:, :, state_dim:2*state_dim]
            
            pred_new_state_seq = model(states_seq, action_seq)
            # if (epoch % 2 == 0) and (batch_count >1) and (batch_count <4):
            #     for i in range(state_dim):
            #         plt.plot(pred_new_state_seq[:,:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
            #         plt.plot(new_state_seq[:,:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
            #         plt.show()
                
            total_val_loss += loss_fn(pred_new_state_seq, new_state_seq).item()
            batch_count += 1
    # Compute mean losses
    mean_train_loss = total_train_loss / len(train_loader)
    mean_val_loss = total_val_loss / len(val_loader)
    ep_train_loss.append(mean_train_loss)
    ep_val_loss.append(mean_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")

    # Clear and redraw the plot
    # ax.clear()
    # ax.plot(ep_train_loss, label='Train Loss', color='blue', marker='o')
    # ax.plot(ep_val_loss, label='Val Loss', color='red',marker='o')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.set_title('Training and Validation Loss')
    # ax.legend()
    # ax.grid(True)
    # plt.tight_layout()
    # plt.pause(0.01)  # Pause briefly to allow GUI update

# plt.ioff()  # Turn off interactive mode at the end
# plt.show()

torch.save(model.state_dict(), "modeler_rnn.pth")
