import numpy as np
import matplotlib.pyplot as plt
from actor_transformer import VehicleActor
from modeler_transformer import VehicleModeler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from utilites import LoadData, GetData

###load data into batches
seq_len = 50       # sequence length for transformer
batch_size = 128    # number of sequences per batch
num_epochs = 40    # how many passes over the dataset
train_loader, val_loader, state_dim, error_dim, action_dim = LoadData("offline_data/filename1.csv", 0.2, batch_size, seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
#                         hidden_dim = 256, rnn_layers = 2,
#                         ).to(device)
model = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
                        d_model = 256, nhead = 2, num_layers=2,
                        ).to(device)
loss_fn = nn.MSELoss(reduction = 'mean')

# Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
#                  hidden_dim = 256, rnn_layers = 3,
#                  ).to(device)

Vmodel = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                 d_model = 128, nhead = 2, num_layers = 2,
                 ).to(device)
Vmodel.load_state_dict(torch.load('modeler_rnn.pth', map_location=device))

for param in Vmodel.parameters():
    param.requires_grad = False

Vmodel.train()
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, amsgrad = True)


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
        initial_state, initial_error_state, \
        state_seq, _, error_state_seq, _, _, _=GetData(batch, state_dim, error_dim, action_dim)

        actor_states= torch.cat([initial_state, initial_error_state], dim = 2)
        pred_actions= model.forward(actor_states, seq_len)  # Your model takes (state, error) as inputs
        # print(pred_actions.shape)
        
        # with torch.no_grad():
        pred_states = Vmodel(initial_state, pred_actions)

        #computing predicted state error as the loss
        #x* - x = e  so x* = e+x
        e_depth = error_state_seq[:,:,0]
        e_u = error_state_seq[:,:,5]

        e_cos_pitch = error_state_seq[:, :, 1]
        e_sin_pitch = error_state_seq[:, :, 2]
        e_pitch = torch.atan2(e_sin_pitch, e_cos_pitch)

        e_cos_yaw = error_state_seq[:, :, 3]
        e_sin_yaw = error_state_seq[:, :, 4]
        e_yaw = torch.atan2(e_sin_yaw, e_cos_yaw)

        e_controlled_state = torch.stack([e_depth, e_pitch, e_yaw, e_u ], dim = -1)


        m_cos_yaw = state_seq[:, :, 3]
        m_sin_yaw = state_seq[:, :, 4]
        m_yaw = torch.atan2(m_sin_yaw, m_cos_yaw)


        m_depth = state_seq[:,:,0]

        m_cos_pitch = state_seq[:, :, 1]
        m_sin_pitch = state_seq[:, :, 2]
        m_pitch = torch.atan2(m_sin_pitch, m_cos_pitch)

        m_cos_yaw = state_seq[:, :, 3]
        m_sin_yaw = state_seq[:, :, 4]
        m_yaw = torch.atan2(m_sin_yaw, m_cos_yaw)

        m_u = state_seq[:, : ,5]

        m_controlled_state = torch.stack([m_depth, m_pitch, m_yaw, m_u ], dim = -1)

        desired_states = e_controlled_state + m_controlled_state

        ##Predicted state
        p_depth = pred_states[:,:,0]
        p_cos_pitch = pred_states[:, :, 1]
        p_sin_pitch = pred_states[:, :, 2]
        p_pitch = torch.atan2(p_sin_pitch, p_cos_pitch)

        p_cos_yaw = pred_states[:, :, 3]
        p_sin_yaw = pred_states[:, :, 4]
        p_yaw = torch.atan2(p_sin_yaw, p_cos_yaw)

        p_u = pred_states[:, : ,5]

        pred_controlled_state = torch.stack([p_depth, p_pitch, p_yaw, p_u ], dim = -1)

        #error between desired and the predicted states
        pred_e = desired_states - pred_controlled_state

        #redo the angle different for wrapping problem
        diff_pitch = pred_e[:,:,1]
        diff_yaw = pred_e[:,:,2]

        #solve the wrapping problem
        diff_yaw = (diff_yaw + np.pi) % (2 * np.pi) - np.pi
        diff_pitch = (diff_pitch + np.pi) % (2 * np.pi) - np.pi

        pred_e[:,:,1] = diff_pitch
        pred_e[:,:,2] = diff_yaw

        loss = torch.sum(pred_e**2)
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

        if (epoch % 10 == 0) and (batch_count >1) and (batch_count <3):
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
