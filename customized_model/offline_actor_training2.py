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
seq_len = 50       # sequence length for transformer
batch_size = 128    # number of sequences per batch
num_epochs = 20    # how many passes over the dataset
train_loader, val_loader, state_dim, error_dim, action_dim = LoadData("offline_data/filename1.csv", 0.2, batch_size, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

max_action = torch.tensor([0.6, 0.6, 0.5, 0.5]).to(device)

# model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
#                         hidden_dim = 64, rnn_layers = 2,
#                         ).to(device)
model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
                        d_model = 128, nhead = 8, num_layers=2, max_action= max_action, dropout=0.0
                        ).to(device)
# model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
#                         hidden_dim = 256, seq_len = seq_len, num_layers =3, max_action = 0.7
#                         ).to(device)
# loss_fn = nn.MSELoss(reduction = 'mean')

# Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
#                  hidden_dim = 128, rnn_layers = 2,
#                  ).to(device)

Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim, dropout=0.0,
                 d_model = 128, nhead = 8, num_layers = 2,
                 ).to(device)
Vmodel.load_state_dict(torch.load('offline_model/modeler.pth', map_location=device))

for param in Vmodel.parameters():
    param.requires_grad = False

Vmodel.train()
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

torch.autograd.set_detect_anomaly(True)
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
        
        #zero depth for reference
        zero_depth_initial_state = initial_state.clone().detach()
        zero_depth_initial_state[:,:,0] = 0
        actor_states= torch.cat([zero_depth_initial_state, initial_error_state], dim = 2)

        ###########################MPPI##############################################
        # num_samples = 20
        # mean_action_seq = torch.zeros(batch_size, seq_len, action_dim, device=device)
        # noise = 0.1 * torch.randn(batch_size, num_samples, seq_len, action_dim, device=device)
        # current_state_expanded = zero_depth_initial_state.unsqueeze(1)
        # initial_state_expanded = initial_state.unsqueeze(1)
        # set_point_expanded = initial_setpoint_state.unsqueeze(1)

        # current_state_expanded = current_state_expanded.repeat(1, num_samples, 1, 1)
        # initial_state_expanded = initial_state_expanded.repeat(1, num_samples, 1, 1)
        # set_point_expanded = set_point_expanded.repeat(1, num_samples,1 ,1)

        # # print(set_point_expanded.shape)
        # # print(initial_state_expanded.shape)
        
        # action_samples = mean_action_seq.unsqueeze(1) + noise  # (B, N, H, D)

        # action_samples = torch.clamp(action_samples, -max_action, max_action)
        # print(action_samples.shape)

        # flat_actions = action_samples.view(-1, seq_len, action_dim)

        # flat_states = current_state_expanded.view(-1, zero_depth_initial_state.shape[-1])
        # flat_states = flat_states.unsqueeze(1)
        

        # flat_initial_states = initial_state_expanded.view(-1, zero_depth_initial_state.shape[-1])
        # flat_initial_states = flat_initial_states.unsqueeze(1)

        # flat_setpoint = set_point_expanded.view(-1, initial_setpoint_state.shape[-1])
        # flat_setpoint = flat_setpoint.unsqueeze(1)

        # with torch.no_grad():
        #     s_pred_seq = Vmodel(flat_states, flat_actions)  # (B*N, H, state_dim)

        #     s_pred_seq[:, :, 0] = s_pred_seq[:, :, 0] + flat_initial_states[:, :, 0]

        #     # Extract predicted states
        #     p_depth = s_pred_seq[:, :, ind_s_z]
        #     p_cos_pitch = s_pred_seq[:, :, ind_s_cos_pitch]
        #     p_sin_pitch = s_pred_seq[:, :, ind_s_sin_pitch]
        #     p_cos_yaw = s_pred_seq[:, :, ind_s_cos_yaw]
        #     p_sin_yaw = s_pred_seq[:, :, ind_s_sin_yaw]
        #     p_u = s_pred_seq[:, :, ind_s_u]

        #     future_states = torch.stack([p_depth, p_cos_pitch, p_sin_pitch, p_cos_yaw, p_sin_yaw, p_u], dim=-1)

        #     future_errors = future_states - flat_setpoint

        #     pitch_diff = angular_difference(p_cos_pitch, p_sin_pitch, flat_setpoint[:,:,ind_e_cos_pitch], 
        #                                     flat_setpoint[:,:,ind_e_sin_pitch])
        #     yaw_diff = angular_difference(p_cos_yaw, p_sin_yaw, flat_setpoint[:,:,ind_e_cos_yaw], 
        #                                     flat_setpoint[:,:,ind_e_sin_yaw])

        #     error_diff = torch.stack([future_errors[:,:,0], pitch_diff, yaw_diff, 2*future_errors[:,:,5]], dim=-1)

        #     # Step 1: Compute absolute cost
        #     cost = torch.abs(error_diff).sum(dim=-1)  # shape: (B*N, H)

        #     # Step 2: Reshape to (B, N, H)
        #     cost = cost.view(batch_size, num_samples, seq_len)

        #     # Step 3: Create time-step weights and apply
        #     time_weights = torch.linspace(0.1, 1.0, seq_len).to(device)  # shape: (H,)
        #     time_weights = time_weights.view(1, 1, seq_len)  # reshape to (1, 1, H) for broadcasting
        #     weighted_cost = cost * time_weights  # shape: (B, N, H)

        #     # Step 4: Sum over time for final cost per trajectory
        #     cost_per_traj = weighted_cost.sum(dim=-1)  # shape: (B, N)
        #     weights = torch.softmax(-cost_per_traj / 0.1, dim=1)  # shape: (B, N)
            
        #     weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)

        #     weighted_actions = torch.sum( weights_expanded* action_samples, dim=1)  # shape (B, H, D)
        #     a_seq = weighted_actions

        #  a_seq[:] = torch.max(torch.min(a_seq, max_action), -max_action)
        ###########################Gradien based######################################
        # a_seq = torch.randn(batch_size, seq_len, action_dim).to(device)
        a_seq = (torch.rand(batch_size, seq_len, action_dim) * 2 - 1).to(device)
        
        with torch.no_grad():
            # a_seq_init = model(actor_states, seq_len)

            # Step 2: Make it a leaf tensor for optimization
            # a_seq = torch.nn.Parameter(a_seq_init.clone().detach(), requires_grad=True)
            a_seq[:] = torch.max(torch.min(a_seq, max_action), -max_action)
        a_seq.requires_grad_()
        optimizer = torch.optim.Adam([a_seq], lr=1e-1)
        for i in range(50):
            s_pred_seq = Vmodel(zero_depth_initial_state, a_seq)
            s_pred_seq = s_pred_seq.clone()
            s_pred_seq[:, :, 0] = s_pred_seq[:, :, 0] + initial_state[:, :, 0]

            # Extract predicted states
            p_depth = s_pred_seq[:, :, ind_s_z]
            p_cos_pitch = s_pred_seq[:, :, ind_s_cos_pitch]
            p_sin_pitch = s_pred_seq[:, :, ind_s_sin_pitch]
            p_cos_yaw = s_pred_seq[:, :, ind_s_cos_yaw]
            p_sin_yaw = s_pred_seq[:, :, ind_s_sin_yaw]
            p_u = s_pred_seq[:, :, ind_s_u]

            future_states = torch.stack([p_depth, p_cos_pitch, p_sin_pitch, p_cos_yaw, p_sin_yaw, p_u], dim=-1)

            final_error = future_states[:, -1, :] - initial_setpoint_state
            
            future_errors = future_states - initial_setpoint_state

            pitch_diff = angular_difference(p_cos_pitch, p_sin_pitch, initial_setpoint_state[:,:,ind_e_cos_pitch], 
                                            initial_setpoint_state[:,:,ind_e_sin_pitch])
            yaw_diff = angular_difference(p_cos_yaw, p_sin_yaw, initial_setpoint_state[:,:,ind_e_cos_yaw], 
                                            initial_setpoint_state[:,:,ind_e_sin_yaw])

            error_diff = torch.stack([future_errors[:,:,0], 2*pitch_diff, yaw_diff, 2*future_errors[:,:,5]], dim=-1)


            weights = torch.linspace(0.1, 1.0, seq_len).to(device)
            weights = weights.view(1, seq_len, 1)
            # print(weights.shape)
            weighed_error = error_diff*weights
            # weighted_loss = (error_term.sum(dim=-1).pred_e.shape[0] * weights).mean()
            weighted_loss = torch.sum(abs(weighed_error))

            # goal_loss = torch.sum(abs(error_diff))

            # Smoothness loss
            jerk = error_diff[:,2:,:] - 2* error_diff[:,1:-1,:] + error_diff[:,:-2,:]
            jerk_loss = torch.sum(abs(jerk))   

            # Total loss and backward
            total_loss = weighted_loss + 1.0* jerk_loss
        
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Clamp actions in-place, preserving computation graph
            with torch.no_grad():
                a_seq[:] = torch.max(torch.min(a_seq, max_action), -max_action)

            # print(f"Step {i}: Loss = {total_loss.item():.4f}")
        ####################################################################################
        #plot predicted states final errors
        # pre_e_flat = future_errors.reshape(-1, 4)
        # for i in range(4):
        #     dd = error_diff[1,:,i].detach().cpu().numpy() 
        #     # plt.plot(plot_predict_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
        #     plt.plot(abs(dd), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
        #     # plt.plot(a_seq[1,:,i].detach().cpu().numpy(),label='Label (optional)', color='red', linestyle='-', marker='o')
        #     plt.show()

        ##actor
        pred_actions= model.forward(actor_states, seq_len)  # Your model takes (state, error) as inputs
        diff = pred_actions - a_seq
        actor_loss = torch.sum(diff**2)
        # print(loss.item())
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()


        # #predict with actions
        # s_pred_seq = Vmodel(zero_depth_initial_state, pred_actions)
        # s_pred_seq = s_pred_seq.clone()
        # s_pred_seq[:, :, 0] = s_pred_seq[:, :, 0] + initial_state[:, :, 0]

        # # Extract predicted states
        # p_depth = s_pred_seq[:, :, ind_s_z]
        # p_cos_pitch = s_pred_seq[:, :, ind_s_cos_pitch]
        # p_sin_pitch = s_pred_seq[:, :, ind_s_sin_pitch]
        # p_cos_yaw = s_pred_seq[:, :, ind_s_cos_yaw]
        # p_sin_yaw = s_pred_seq[:, :, ind_s_sin_yaw]
        # p_u = s_pred_seq[:, :, ind_s_u]

        # future_states = torch.stack([p_depth, p_cos_pitch, p_sin_pitch, p_cos_yaw, p_sin_yaw, p_u], dim=-1)
        
        # future_errors = future_states - initial_setpoint_state

        # pitch_diff = angular_difference(p_cos_pitch, p_sin_pitch, initial_setpoint_state[:,:,ind_e_cos_pitch], 
        #                                 initial_setpoint_state[:,:,ind_e_sin_pitch])
        # yaw_diff = angular_difference(p_cos_yaw, p_sin_yaw, initial_setpoint_state[:,:,ind_e_cos_yaw], 
        #                                 initial_setpoint_state[:,:,ind_e_sin_yaw])

        # error_diff = torch.stack([future_errors[:,:,0], pitch_diff, yaw_diff, 2*future_errors[:,:,5]], dim=-1)

        # if (epoch % 5 == 0) and (batch_count >1) and (batch_count <3) and (epoch>3):
        #     for i in range(4):
        #         dd = error_diff[1,:,i].detach().cpu().numpy() 
        #         # plt.plot(plot_predict_data[:,i].detach().cpu().numpy(), label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed
        #         plt.plot(dd, label='Label (optional)', color='red', linestyle='-', marker='o')  # Customize as needed

        #         plt.show()

        print(f"batch number = {batch_count}/{len(train_loader)}")
        batch_count += 1
        
        total_train_loss += actor_loss.item()

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
