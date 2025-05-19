import numpy as np
import matplotlib.pyplot as plt
from network.actor_transformer import VehicleActor
# from network.actor_fnn import VehicleActor
from network.modeler_transformer import VehicleModeler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from network.utilites import LoadData, GetData, safe_atan2

###load data into batches
seq_len = 20       # sequence length for transformer
batch_size = 8    # number of sequences per batch
num_epochs = 20    # how many passes over the dataset
train_loader, val_loader, state_dim, error_dim, action_dim = LoadData("offline_data/filename1.csv", 0.2, batch_size, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
                        d_model = 256, nhead = 8, num_layers=3, max_action= 0.7, dropout=0.0
                        ).to(device)

Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 d_model = 128, nhead = 8, num_layers = 3, dropout=0.0
                 ).to(device)

Vmodel.load_state_dict(torch.load('offline_model/modeler.pth', map_location=device))

for param in Vmodel.parameters():
    param.requires_grad = False

Vmodel.eval()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, amsgrad = True)


ep_train_loss = []
ep_val_loss = []

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

plt.ion()  # Turn on interactive mode

fig, axes = plt.subplots(2, 4, figsize=(12, 5))  # 2 rows, 4 columns
ax1, ax2, ax3, ax4 = axes[0]                     # First row
ax21, ax22, ax23, ax24 = axes[1]                 # Second row

for epoch in range(num_epochs):
    
    total_train_loss = 0.0
    total_val_loss = 0.0
    # model.train()
    # Keep Vmodel in evaluation mode since it's not being trained
    # Vmodel.eval()
    batch_count = 0
    for batch in train_loader:
        batch = batch.to(device)  # shape: (batch_size, seq_len, input_dim + action_dim)
        initial_state, initial_error_state, initial_setpoint_state, \
        state_seq, _, error_state_seq, new_error_state_seq, set_point_seq, action_seq, _=GetData(batch, state_dim, error_dim, action_dim)
        
        ############################################################
        ############## model based ######################
        ############################################################
        ##restore desired state from state seq and error_state_seq
        c_depth = initial_setpoint_state[:,:,ind_e_z] 
        c_u = initial_setpoint_state[:,:, ind_e_u]

        c_sin_pitch = initial_setpoint_state[:,:, ind_e_sin_pitch]
        c_cos_pitch = initial_setpoint_state[:,:, ind_e_cos_pitch]

        c_sin_yaw = initial_setpoint_state[:,:, ind_e_sin_yaw]
        c_cos_yaw = initial_setpoint_state[:,:, ind_e_cos_yaw]
        
        for iter in range(100):
            # print(iter)

            zero_depth_initial_state = initial_state.clone()
            zero_depth_initial_state[:,:,0] = 0


            actor_states= torch.cat([zero_depth_initial_state, initial_error_state], dim = 2)
            pred_actions= model.forward(actor_states, seq_len)  # Your model takes (state, error) as inputs
            # print(pred_actions.shape)
            
            # with torch.no_grad():
            pred_states = Vmodel(zero_depth_initial_state, pred_actions)
            pred_states [:,:,0] = pred_states[:,:,0] + initial_state[:,:,0]

            ##Predicted state
            p_depth = pred_states[:,:,ind_s_z]
            p_u = pred_states[:, : ,ind_s_u]

            p_cos_pitch = pred_states[:, :, ind_s_cos_pitch]
            p_sin_pitch = pred_states[:, :, ind_s_sin_pitch]

            p_cos_yaw = pred_states[:, :, ind_s_cos_yaw]
            p_sin_yaw = pred_states[:, :, ind_s_sin_yaw]

        
            pred_e_depth = c_depth - p_depth
            pred_e_u     = c_u - p_u
            pred_e_cos_pitch = c_cos_pitch - p_cos_pitch 
            pred_e_sin_pitch = c_sin_pitch - p_sin_pitch
            pred_e_cos_yaw = c_cos_yaw - p_cos_yaw
            pred_e_sin_yaw = c_sin_yaw - p_sin_yaw

            ##actual states we are going for loss compuation

            
            pred_e = torch.stack([pred_e_depth, pred_e_cos_pitch, pred_e_sin_pitch, pred_e_cos_yaw,  pred_e_sin_yaw, pred_e_u ], dim = -1)
            pred_e= torch.cat([initial_error_state, pred_e], dim = 1)

            w = torch.tensor([[1, 10, 10, 5, 5, 5]], dtype=torch.float32)  # shape: [1, 5]
            w = w.unsqueeze(0).to(device)

            pred_e_w = pred_e * w
            #weighted starting from the first one which is zero (current state)
            weights = torch.linspace(0.0, 1.0, seq_len+1).to(device)
            weights = weights.view(1, seq_len+1, 1)
            error_term = pred_e_w**2
            # print(weights.shape)
            # print(error_term.shape)

            weighted_loss = torch.mean(error_term*weights)

            jerk = pred_e[:,2:,:] - 2* pred_e[:,1:-1,:] + pred_e[:,:-2,:]
            jerk_loss = torch.mean(jerk **2)   

            delt_state = pred_e[:,1:,:] - pred_e[:,:-1,:]
            delta_state_loss = torch.mean(delt_state **2) 

            #total energy
            energy_loss = torch.mean(pred_actions  **2)

            total_loss = 0.1*jerk_loss + weighted_loss + 0.1*delta_state_loss + 0.1*energy_loss
            # print(total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            for name, param in Vmodel.named_parameters():
                if param.requires_grad:
                    print(f"{name} can be updated")  # Should not print anything
            optimizer.step()
            

            ##display
            #desired - predition
            pred_e_cos_pitch = c_cos_pitch*p_cos_pitch + c_sin_pitch *p_sin_pitch
            pred_e_sin_pitch = c_sin_pitch*p_cos_pitch - c_cos_pitch*p_sin_pitch
            pred_e_cos_yaw = c_cos_yaw*p_cos_yaw + c_sin_yaw *p_sin_yaw
            pred_e_sin_yaw = c_sin_yaw*p_cos_yaw - c_cos_yaw*p_sin_yaw
            pred_e_pitch = safe_atan2(pred_e_sin_pitch, pred_e_cos_pitch)
            pred_e_yaw = safe_atan2(pred_e_sin_yaw, pred_e_cos_yaw)

            pred_display = torch.stack([pred_e_depth, pred_e_pitch, pred_e_yaw, pred_e_u ], dim = -1)

            ##add current error into the pred_e
            e_depth_0 = initial_error_state[:,:,ind_e_z]
            e_u_0 = initial_error_state[:,:, ind_e_u]
            e_pitch_0 = torch.atan2(initial_error_state[:,:,ind_e_sin_pitch], initial_error_state[:,:,ind_e_cos_pitch])
            e_yaw_0 = torch.atan2(initial_error_state[:,:,ind_e_sin_yaw], initial_error_state[:,:,ind_e_cos_yaw])
            e_0 = torch.stack([e_depth_0, e_pitch_0, e_yaw_0, e_u_0 ], dim = -1)

            ##actual states we are going for loss compuation
            considered_error_state= torch.cat([e_0, pred_display], dim = 1)
            
            
            ########visualization
           
            fig.suptitle(f"iteration: {iter}, batch no: {batch_count}, epoch: {epoch}", fontsize=16)
            # ax4.set_ylim(-3, 3)
            # ax3.set_ylim(-0.2, 0.2)
            # ax2.set_ylim(-3.2, 3.2)
            # ax1.set_ylim(-1, 1)
            ax4.set_title("depth error")
            ax3.set_title("pitch error")
            ax2.set_title("yaw error")
            ax1.set_title("surge error")

            ax21.set_ylim(-1, 1)
            ax22.set_ylim(-1, 1)
            ax23.set_ylim(-1, 1)
            ax24.set_ylim(-1, 1)
            ax21.set_title("surge")
            ax22.set_title("sway")
            ax23.set_title("heave stern")
            ax24.set_title("heave bow")
            #add  current error

            dd =considered_error_state.detach().cpu().numpy() 
            axes = [ax4, ax3, ax2, ax1]
            for i, ax in enumerate(axes):
                ax.plot(dd[1,:,i], label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
                ax.plot(dd[1,0,i], label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
                ax.axhline(y=0, color='black', linewidth=2.0, zorder=5)  # You can adjust color and width

            axes = [ax21, ax22, ax23, ax24]
            for i, ax in enumerate(axes):
                ax.plot(pred_actions[1,:,i].detach().cpu().numpy() , label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
                ax.axhline(y=0, color='black', linewidth=2.0, zorder=5)  # You can adjust color and width
            
            plt.pause(0.1)

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax21.clear()
            ax22.clear()
            ax23.clear()
            ax24.clear()
            # pre_e_flat = pred_e.reshape(-1, 4)
            # if (epoch % 2 == 0) and (batch_count >1) and (batch_count <3):
            #     for i in range(4):
            #         dd = pre_e_flat[:,i].detach().cpu().numpy() 
            #         # plt.plot(plot_desired_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
            #         # plt.plot(plot_predict_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
            #         plt.plot(abs(dd), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

            #         plt.show()
        batch_count += 1
        
        total_train_loss += total_loss.item()

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

    torch.save(model.state_dict(), "offline_model/actor.pth")
