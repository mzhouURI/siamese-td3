import numpy as np
import matplotlib.pyplot as plt
from network.modeler_transformer import VehicleModeler
# from network.modeler_rnn import VehicleModeler
# from network.modeler_drnn import VehicleModeler

import torch
import torch.nn as nn
from network.utilites import LoadData, GetData


###load data into batches
seq_len = 100       # sequence length for transformer
batch_size =2    # number of sequences per batch
train_loader, val_loader, state_dim, error_dim, action_dim = LoadData("offline_data/filename1.csv", 0.1, batch_size, seq_len)


##make modlayernorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
#                  hidden_dim = 128, rnn_layers = 2,
#                  ).to(device)

model = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 d_model = 128, nhead = 8, num_layers = 3, dropout=0.0
                 ).to(device)
model.load_state_dict(torch.load('offline_model/modeler.pth', map_location=device))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = True)
loss_fn = nn.MSELoss(reduction = 'sum')


ep_train_loss = []
ep_val_loss = []

ind_s_cos_pitch = 3
ind_s_sin_pitch =4
ind_s_cos_yaw =5
ind_s_sin_yaw = 6
ind_s_u = 7
ind_s_z = 0

plt.ion()  # Turn on interactive mode

fig, axes = plt.subplots(2, 4, figsize=(12, 5))  # 2 rows, 4 columns
ax1, ax2, ax3, ax4 = axes[0]                     # First row
ax21, ax22, ax23, ax24 = axes[1]                 # Second row

model.eval()

batch_count = 0

for batch in train_loader:
    batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)

    initial_state, _, _, _, new_state_seq, _, _, _, action_seq, _=GetData(batch, state_dim, error_dim, action_dim)

    zero_depth_initial_state = initial_state.clone()
    #zero depth for reference
    zero_depth_initial_state[:,:,0] = 0
    # print(zero_depth_initial_state.shape)

    pred_new_state_seq = model.forward(zero_depth_initial_state, action_seq)  # Your model takes (state, error) as inputs
    # print(pred_new_state_seq.shape)
    pred_new_state_seq [:,:,0] = pred_new_state_seq[:,:,0] + initial_state[:,:,0]

    pred_pitch = torch.atan2(pred_new_state_seq[:,:,ind_s_sin_pitch], pred_new_state_seq[:,:,ind_s_cos_pitch])
    pred_yaw = torch.atan2(pred_new_state_seq[:,:,ind_s_sin_yaw], pred_new_state_seq[:,:,ind_s_cos_yaw])
    actual_pitch = torch.atan2(new_state_seq[:,:,ind_s_sin_pitch], new_state_seq[:,:,ind_s_cos_pitch])
    actual_yaw = torch.atan2(new_state_seq[:,:,ind_s_sin_yaw], new_state_seq[:,:,ind_s_cos_yaw])

    # print(pred_pitch.shape)
    pred_controlled_state = torch.stack([pred_new_state_seq[:,:,0], pred_pitch, pred_yaw, pred_new_state_seq[:,:, ind_s_u]], dim = -1) 
    actual_controlled_state = torch.stack([new_state_seq[:,:,0], actual_pitch, actual_yaw, new_state_seq[:,:, ind_s_u]], dim = -1) 
    batch_count += 1  
        # print(f"batch no: {batch_count}/{len(train_loader)}")

    fig.suptitle(f"batch no: {batch_count}/{len(train_loader)}", fontsize=16)

    ax1.set_title("depth")
    ax2.set_title("pitch")
    ax3.set_title("yaw")
    ax4.set_title("surge")
    ax21.set_title("surge")
    ax22.set_title("sway")
    ax23.set_title("heave stern")
    ax24.set_title("heave bow")

    ax1.set_ylim(-7, 1)
    # ax2.set_ylim(-0.3, 0.3)
    ax3.set_ylim(-3.2, 3.2)
    ax4.set_ylim(-1, 1)
    ax21.set_ylim(-1, 1)
    ax22.set_ylim(-1, 1)
    ax23.set_ylim(-1, 1)
    ax24.set_ylim(-1, 1)
    
    axes = [ax1, ax2, ax3, ax4]
    for i, ax in enumerate(axes):
        ax.plot(actual_controlled_state[1,:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
        ax.plot(pred_controlled_state[1,:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

    axes = [ax21, ax22, ax23, ax24]
    for i, ax in enumerate(axes):
        ax.plot(action_seq[1,:,i].detach().cpu().numpy() , label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

    plt.pause(0.1)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax21.clear()
    ax22.clear()
    ax23.clear()
    ax24.clear()     

    

