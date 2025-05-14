import numpy as np
import matplotlib.pyplot as plt
from actor import VehicleActor
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
pitch = data[:-1,17]
yaw = data[:-1,18]
cos_pitch = np.cos(pitch)
sin_pitch = np.sin(pitch)
cos_yaw = np.cos(yaw)
sin_yaw = np.sin(yaw)


# Remove pitch and yaw columns from `states`
base_states = data[:-1, [15, 19, 23, 24]]
# Trim cos/sin arrays to match states shape (one less row due to data[:-1])
# Stack new states: [x, cos_pitch, sin_pitch, cos_yaw, sin_yaw, z, vx, vy]
states = np.column_stack((base_states[:, 0],   # col 15 (x)
                          cos_pitch,
                          sin_pitch,
                          cos_yaw,
                          sin_yaw,
                          base_states[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                         ))

#depth, pitch, yaw, surge
error_states = data[:-1, [3,5,6,7]]


current_action = data[:-1, -4:]
action_seq = data[1:, -4:]

pitch = data[1:,17]
yaw = data[1:,18]
cos_pitch = np.cos(pitch)
sin_pitch = np.sin(pitch)
cos_yaw = np.cos(yaw)
sin_yaw = np.sin(yaw)
base_state_seq = data[1:, [15, 19, 23, 24]]
state_seq = np.column_stack((base_state_seq[:, 0],   # col 15 (x)
                          cos_pitch,
                          sin_pitch,
                          cos_yaw,
                          sin_yaw,
                          base_state_seq[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                         ))
# state_seq = data[1:, [15,17,18,19,23,24]]

state_dim = states.shape[1]
error_dim = error_states.shape[1]
action_dim = current_action.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VehicleActor(state_dim = state_dim, action_dim = action_dim,
                        hidden_dim = 256, rnn_layers = 2,
                        ).to(device)

loss_fn = nn.MSELoss(reduction = 'mean')


Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 hidden_dim = 256, rnn_layers = 2,
                 ).to(device)
Vmodel.load_state_dict(torch.load('modeler_rnn.pth', map_location=device))

for param in Vmodel.parameters():
    param.requires_grad = False
Vmodel.train()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad = True)

# Assuming error_states, states, and actions are already extracted from `data`
training_data = np.hstack((states, state_seq, error_states, current_action, action_seq))


ep_loss = []
batch_size = 128    # number of sequences per batch
num_epochs = 40    # how many passes over the dataset
seq_len = 20       # sequence length for transformer


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

# plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

ep_train_loss = []
ep_val_loss = []

for epoch in range(num_epochs):
    
    total_train_loss = 0.0
    total_val_loss = 0.0
    # model.train()
    # Keep Vmodel in evaluation mode since it's not being trained
    # Vmodel.eval()
    batch_count = 0
    for batch in train_loader:
        batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)


        # state = batch[:, 0, :state_dim].clone().detach().requires_grad_()
        # action = batch[:, 0, -2*action_dim:-action_dim].clone().detach().requires_grad_()

        state = batch[:, 0, :state_dim].clone().detach().requires_grad_()
        # action = batch[:, 0, -2*action_dim:-action_dim].clone().detach().requires_grad_()

        state = state.unsqueeze(1)
        # action = action.unsqueeze(1)

        new_state_seq = batch[:, :, state_dim:2*state_dim]
        action_seq = batch[:, :, -action_dim:]
        
        pred_actions= model.forward(state, 20)  # Your model takes (state, error) as inputs
        # print(pred_actions.shape)
        
        # with torch.no_grad():
        pred_states = Vmodel(state, pred_actions)

        #computing predicted state error as the loss
        #x* - x = e  so x* = e+x
        states = batch [:, :, :state_dim]
        state_errors = batch[:, :, 2*state_dim:2*state_dim+4]

        m_depth = states[:,:,0]
        m_cos_pitch = states[:, :, 1]
        m_sin_pitch = states[:, :, 2]
        m_pitch = torch.atan2(m_sin_pitch, m_cos_pitch)

        m_cos_yaw = states[:, :, 3]
        m_sin_yaw = states[:, :, 4]
        m_yaw = torch.atan2(m_sin_yaw, m_cos_yaw)

        m_u = states[:, : ,5]

        

        m_controlled_state = torch.stack([m_depth, m_pitch, m_yaw, m_u ], dim = -1)

        desired_states = state_errors + m_controlled_state

        p_depth = pred_states[:,:,0]
        p_cos_pitch = pred_states[:, :, 1]
        p_sin_pitch = pred_states[:, :, 2]
        p_pitch = torch.atan2(p_sin_pitch, p_cos_pitch)

        p_cos_yaw = pred_states[:, :, 3]
        p_sin_yaw = pred_states[:, :, 4]
        p_yaw = torch.atan2(p_sin_yaw, p_cos_yaw)

        p_u = pred_states[:, : ,5]

        pred_controlled_state = torch.stack([p_depth, p_pitch, p_yaw, p_u ], dim = -1)

        # print(pred_controlled_state.shape)

        pred_e = desired_states - pred_controlled_state

        diff_pitch = pred_e[:,:,1]
        diff_yaw = pred_e[:,:,2]

        diff_yaw = (diff_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pitch = (diff_pitch + np.pi) % (2 * np.pi) - np.pi

        pred_e[:,:,1] = diff_pitch
        pred_e[:,:,2] = diff_yaw

        loss = torch.mean(pred_e**2)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        for name, param in Vmodel.named_parameters():
            if param.requires_grad:
                print(f"{name} can be updated")  # Should not print anything
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: min = {param.grad.min().item():.6f}, max = {param.grad.max().item():.6f}")
        
        optimizer.step()

        # plot_desired_data = desired_states.reshape(-1, 4)
        # plot_predict_data = pred_controlled_state.reshape(-1, 4)
        pre_e_flat = pred_e.reshape(-1, 4)
        # print(pred_e.shape)

        if (epoch % 5 == 0) and (batch_count >1) and (batch_count <3):
            for i in range(4):
                dd = pre_e_flat[:,i].detach().cpu().numpy() 
                # plt.plot(plot_desired_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
                # plt.plot(plot_predict_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
                plt.plot(abs(dd), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

                plt.show()

        batch_count += 1
        
        total_train_loss += loss.item()

    # --- VALIDATION ---
    # model.eval()
    # batch_count = 1
    # # print(epoch)
    # with torch.no_grad():
    #     for batch in val_loader:
    #         batch = batch.to(device)

    #         state = batch [:, 0, :state_dim]
    #         action =batch [:, 0, -2*action_dim:-action_dim]
    
    #         state = state.unsqueeze(1)
    #         action = action.unsqueeze(1)
            
    #         new_state_seq = batch[:, :, state_dim:2*state_dim]
    #         action_seq = batch[:, :, -action_dim:]

    #         pred_actions= model.forward(state, action, 20)  # Your model takes (state, error) as inputs
    #         # print(pred_actions.shape)

    #         if (epoch % 5 == 0) and (batch_count >1) and (batch_count <3):
    #             for i in range(action_dim):
                    # flat_pred_action = pred_actions.reshape(-1, 4)
                    # flat_action = action_seq.reshape(-1, 4)

                    # plt.plot(flat_pred_action[:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
                    # plt.plot(flat_action[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
                    # plt.show()
                
    #         total_val_loss += loss_fn(pred_actions, action_seq).item()
    #         batch_count += 1
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

torch.save(model.state_dict(), "actor_rnn.pth")
