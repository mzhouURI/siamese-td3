import numpy as np
import matplotlib.pyplot as plt
from siamese import SiamesePoseControlNet, OnlineTrainer
import torch

def get_batches(error_states, states, actions, indices, batch_size):
    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]

        batch_error_states = error_states[batch_indices]
        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]

        yield batch_error_states, batch_states, batch_actions

data = np.loadtxt('data/filename1.csv', delimiter=',')
# 0:12: t, ex, ey, ez, e_roll, e_pitch, e_yaw, e_u, e_v, e_w, e_p, e_q, e_r
#13-15: x,y,z,
#16:18: roll, pitch, yaw, 
#19:24 u,v,w,p,q,r
error_states = data[:, [3,5,6,7]]
states = data[:, 15:25] 
actions = data[:, -4:]

# print(error_states.shape)
# print(states.shape)

# plt.plot(actions[:,3], label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
# plt.show()

# print(states.shape[1])
model = SiamesePoseControlNet(current_pose_dim = states.shape[1], goal_pose_dim = error_states.shape[1], latent_dim = 64, thruster_num = actions.shape[1])
trainer = OnlineTrainer(model, lr=1e-5)

# Assuming error_states, states, and actions are already extracted from `data`
training_data = np.hstack((error_states, states, actions))

# rows_with_nan = np.any(np.isnan(training_data), axis=1)
# nan_row_indices = np.where(rows_with_nan)[0]
# print("Rows with NaN:", nan_row_indices)

# exit()
# print(training_data[1,:])
batch_size = 64  # Choose a batch size
ep_loss = []
for i in range(50):
    # Shuffle the data by generating a random permutation of indices
    indices = np.random.permutation(len(training_data))

    # Example of iterating through the batches
    loss_log = []

    for batch_error_states, batch_states, batch_actions in get_batches(error_states, states, actions, indices, batch_size):

        # Here you can pass the batches to your training function
        # print(batch_error_states.shape, batch_states.shape, batch_actions.shape)

        batch_error_states_tensor = torch.tensor(batch_error_states, dtype=torch.float32)
        batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32)
        batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.float32)

        # print(batch_error_states_tensor.shape)
        # print(batch_states_tensor.shape)

        thrust_pred_batch = model.forward(batch_states_tensor, batch_error_states_tensor)

        loss = trainer.train(thrust_pred_batch, batch_actions_tensor, batch_error_states_tensor)
        loss_log.append(loss)
        # print(thrust_pred_batch)
        # print(batch_actions_tensor)

    print(np.mean(loss_log))
    ep_loss.append(np.mean(loss_log))

trainer.save_model()
plt.plot(ep_loss, label='Label (optional)', color='blue', linestyle='-', marker='o')  # Customize as needed
plt.show()