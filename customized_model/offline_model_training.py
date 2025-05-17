import numpy as np
import matplotlib.pyplot as plt
from network.modeler_transformer import VehicleModeler
import torch
import torch.nn as nn
from network.utilites import LoadData, GetData


###load data into batches
seq_len = 100       # sequence length for transformer
batch_size =32    # number of sequences per batch
num_epochs = 20    # how many passes over the dataset
train_loader, val_loader, state_dim, error_dim, action_dim = LoadData("offline_data/filename1.csv", 0.2, batch_size, seq_len)


##make modlayernorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
#                  hidden_dim = 128, rnn_layers = 2,
#                  ).to(device)

model = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 d_model = 256, nhead = 8, num_layers = 2,
                 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad = True)
loss_fn = nn.MSELoss(reduction = 'sum')

# plt.ion()  # Turn on interactive mode

ep_train_loss = []
ep_val_loss = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_val_loss = 0.0


    for batch in train_loader:
        batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)

        initial_state, _, _, new_state_seq, _, _, action_seq, _=GetData(batch, state_dim, error_dim, action_dim)

        zero_depth_initial_state = initial_state.clone()
        #zero depth for reference
        zero_depth_initial_state[:,:,0] = 0
        # print(zero_depth_initial_state.shape)

        pred_new_state_seq = model.forward(zero_depth_initial_state, action_seq)  # Your model takes (state, error) as inputs
        # print(pred_new_state_seq.shape)
        pred_new_state_seq [:,:,0] = pred_new_state_seq[:,:,0] + initial_state[:,:,0]
    
        loss = nn.MSELoss(reduction="sum")(new_state_seq, pred_new_state_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    # --- VALIDATION ---
    model.eval()
    batch_count = 1

    # with torch.no_grad():
    #     for batch in val_loader:
    #         batch = batch.to(device)
    #         initial_state, _, _, new_state_seq, _, _, action_seq, _=GetData(batch, state_dim, error_dim, action_dim)

    #         zero_depth_initial_state = initial_state.clone()
    #         zero_depth_initial_state[:,:,0] = 0

    #         pred_new_state_seq = model(zero_depth_initial_state, action_seq)
    #         #add depth back to prediction
    #         pred_new_state_seq [:,:,0] = pred_new_state_seq[:,:,0] + initial_state[:,:,0]
    #         #compare predicted depth to the actual depth
    #         total_val_loss += nn.MSELoss(reduction="sum")(new_state_seq, pred_new_state_seq).item()
    #         batch_count += 1

    #         #display
    #         if (epoch % 10 == 0) and (batch_count >1) and (batch_count <3) and (epoch>3):
    #         # if (epoch >35) and (batch_count >1) and (batch_count <3):

    #             for i in range(state_dim):
    #                 flat_pred = pred_new_state_seq.reshape(-1, state_dim)
    #                 flat_actual = new_state_seq.reshape(-1, state_dim)
    #                 plt.plot(flat_pred[:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='None', marker='o')  # Customize as needed
    #                 plt.plot(flat_actual[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='None', marker='o')  # Customize as needed
    #                 plt.show()
    #             #pich and row angle compare
    #             fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))  # Adjust figsize as needed
    #             pre_roll = torch.atan2(flat_pred[:,2], flat_pred[:,1])
    #             actual_roll = torch.atan2(flat_actual[:,2], flat_actual[:,1])
    #             ax1.plot(pre_roll.detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='None', marker='o')  # Customize as needed
    #             ax1.plot(actual_roll.detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='None', marker='o')  # Customize as needed

    #             pre_pitch = torch.atan2(flat_pred[:,4], flat_pred[:,3])
    #             actual_pitch = torch.atan2(flat_actual[:,4], flat_actual[:,3])
    #             ax2.plot(pre_pitch.detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='None', marker='o')  # Customize as needed
    #             ax2.plot(actual_pitch.detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='None', marker='o')  # Customize as needed
                
    #             pre_yaw = torch.atan2(flat_pred[:,6], flat_pred[:,5])
    #             actual_yaw = torch.atan2(flat_actual[:,6], flat_actual[:,5])
    #             ax3.plot(pre_yaw.detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='None', marker='o')  # Customize as needed
    #             ax3.plot(actual_yaw.detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='None', marker='o')  # Customize as needed
    #             plt.show()

            
    # Compute mean losses
    mean_train_loss = total_train_loss / len(train_loader)
    mean_val_loss = total_val_loss / len(val_loader)
    ep_train_loss.append(mean_train_loss)
    ep_val_loss.append(mean_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")

    # Clear and redraw the plot
#     ax.clear()
#     ax.plot(ep_train_loss, label='Train Loss', color='blue', marker='o')
#     ax.plot(ep_val_loss, label='Val Loss', color='red',marker='o')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title('Training and Validation Loss')
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     plt.pause(0.01)  # Pause briefly to allow GUI update

# plt.ioff()  # Turn off interactive mode at the end
# plt.show()

    torch.save(model.state_dict(), "offline_model/modeler.pth")
