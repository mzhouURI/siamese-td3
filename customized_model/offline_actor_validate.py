import numpy as np
import matplotlib.pyplot as plt
from network.actor_transformer import VehicleActor
# from network.actor_fnn import VehicleActor
from network.modeler_transformer import VehicleModeler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from network.utilites import LoadData, GetData, safe_atan2, angular_difference

###load data into batches
seq_len = 30       # sequence length for transformer
action_len = 10
batch_size = 2    # number of sequences per batch
num_epochs = 20    # how many passes over the dataset
filenames = ["offline_data/filename2.csv", "offline_data/filename3.csv"]

train_loader, val_loader, state_dim, error_dim, action_dim = LoadData(filenames, 0.1, batch_size, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

max_action = torch.tensor([0.6, 0.6, 0.5, 0.5]).to(device)

# model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
#                         hidden_dim = 64, rnn_layers = 2,
#                         ).to(device)
max_action = torch.tensor([0.7, 0.6, 0.5, 0.5])  # example per-dimension limits

actor = VehicleActor(state_dim = state_dim, error_dim = error_dim, action_dim = action_dim,
                        d_model = 256, nhead = 8, num_layers=3, max_action= max_action.to(device), dropout=0.05
                        ).to(device)
actor.load_state_dict(torch.load('offline_model/actor.pth', map_location=device))

Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 d_model = 256, nhead = 8, num_layers = 3, dropout=0.0
                 ).to(device)

Vmodel.load_state_dict(torch.load('offline_model/modeler.pth', map_location=device))

Vmodel.eval()
actor.eval()


ind_e_cos_pitch = 1
ind_e_sin_pitch =2
ind_e_cos_yaw =3
ind_e_sin_yaw = 4
ind_e_u = 5
ind_e_z = 0


ind_s_cos_pitch = 3
ind_s_sin_pitch =4
ind_s_cos_yaw =5
ind_s_sin_yaw = 6
ind_s_u = 7
ind_s_z = 0

for batch in val_loader:
    batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)
    initial_state, initial_error_state, initial_setpoint_state, \
    state_seq, _, error_state_seq, new_error_state_seq, set_point_seq, action_seq, _=GetData(batch, state_dim, error_dim, action_dim)
    
    c_depth = initial_setpoint_state[:,:,ind_e_z] 
    c_u = initial_setpoint_state[:,:, ind_e_u]

    c_sin_pitch = initial_setpoint_state[:,:, ind_e_sin_pitch]
    c_cos_pitch = initial_setpoint_state[:,:, ind_e_cos_pitch]

    c_sin_yaw = initial_setpoint_state[:,:, ind_e_sin_yaw]
    c_cos_yaw = initial_setpoint_state[:,:, ind_e_cos_yaw]

    #zero depth for reference
    zero_depth_initial_state = initial_state.clone().detach()
    zero_depth_initial_state[:,:,0] = 0
    actor_states= torch.cat([zero_depth_initial_state, initial_error_state], dim = 2)

    current_action = action_seq[:,0,:].unsqueeze(1)

    pred_actions= actor.forward(zero_depth_initial_state, initial_error_state, current_action, action_len)  # Your model takes (state, error) as inputs

    last_action = pred_actions[:, -1:, :]  # shape: (B, 1, D)
    pad = last_action.repeat(1, seq_len - action_len, 1)
    pred_actions = torch.cat([pred_actions, pad], dim=1)

    pred_states = Vmodel(zero_depth_initial_state, pred_actions)
    pred_states [:,:,0] = pred_states[:,:,0] + initial_state[:,:,0]

    
    ##Predicted state
    p_depth = pred_states[:,:,ind_s_z]
    p_u = pred_states[:, : ,ind_s_u]

    p_cos_pitch = pred_states[:, :, ind_s_cos_pitch]
    p_sin_pitch = pred_states[:, :, ind_s_sin_pitch]

    p_cos_yaw = pred_states[:, :, ind_s_cos_yaw]
    p_sin_yaw = pred_states[:, :, ind_s_sin_yaw]

    depth_diff = p_depth - c_depth
    u_diff = p_u - c_u
    pitch_diff = angular_difference(p_cos_pitch, p_sin_pitch, c_cos_pitch, c_sin_pitch)
    yaw_diff = angular_difference(p_cos_yaw, p_sin_yaw, c_cos_yaw, c_sin_yaw)

    error_data = torch.stack([depth_diff, 2*pitch_diff, yaw_diff, u_diff], dim=-1)

    # pre_e_flat = error_data.reshape(-1, 4)

    for i in range(4):
        # dd = pre_e_flat[:,i].detach().cpu().numpy() 
        # plt.plot(plot_desired_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
        # plt.plot(plot_predict_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
        plt.plot(abs(error_data[1,:,i].detach().cpu().numpy() ), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

        plt.show()
    print("good")