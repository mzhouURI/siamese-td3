import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

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

def LoadData(filename, val_ratio, batch_size, seq_len):
    data = np.loadtxt(filename, delimiter=',')
    # 0:12: t, ex, ey, ez, e_roll, e_pitch, e_yaw, e_u, e_v, e_w, e_p, e_q, e_r
    #13-15: x,y,z,
    #16:18: roll, pitch, yaw, 
    #19:24 u,v,w,p,q,r
    pitch = data[:-1,17]
    yaw = data[:-1,18]
    roll = data[:-1,16]
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    # Remove pitch and yaw columns from `states`
    base_states = data[:-1, [15, 19, 20, 21, 22, 23, 24]]
    # Trim cos/sin arrays to match states shape (one less row due to data[:-1])
    # Stack new states: [x, cos_pitch, sin_pitch, cos_yaw, sin_yaw, z, vx, vy]
    states = np.column_stack((base_states[:, 0],   # col 15 (x)
                            cos_roll,
                            sin_roll,
                            cos_pitch,
                            sin_pitch,
                            cos_yaw,
                            sin_yaw,
                            base_states[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                            ))

    #depth, pitch, yaw, surge
    error_base_states = data[:-1, [3,7]]

    d_pitch = data[1:,5]
    d_yaw = data[1:,6]
    cos_pitch = np.cos(d_pitch)
    sin_pitch = np.sin(d_pitch)
    cos_yaw = np.cos(d_yaw)
    sin_yaw = np.sin(d_yaw)

    error_states = np.column_stack((error_base_states[:, 0],   # col 15 (x)
                            cos_pitch,
                            sin_pitch,
                            cos_yaw,
                            sin_yaw,
                            error_base_states[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                            ))

    current_action = data[:-1, -4:]
    action_seq = data[1:, -4:]

    pitch = data[1:,17]
    yaw = data[1:,18]
    roll = data[1:,16]

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)

    base_state_seq = data[1:,[15, 19, 20, 21, 22, 23, 24]]
    state_seq = np.column_stack((base_state_seq[:, 0],   # col 15 (x)
                            cos_roll,
                            sin_roll,
                            cos_pitch,
                            sin_pitch,
                            cos_yaw,
                            sin_yaw,
                            base_state_seq[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                            ))
    

    error_base_states_seq = data[1:, [3,7]]
    
    d_pitch = data[1:,5]
    d_yaw = data[1:,6]
    cos_pitch = np.cos(d_pitch)
    sin_pitch = np.sin(d_pitch)
    cos_yaw = np.cos(d_yaw)
    sin_yaw = np.sin(d_yaw)

    error_states_seq = np.column_stack((error_base_states_seq[:, 0],   # col 15 (x)
                            cos_pitch,
                            sin_pitch,
                            cos_yaw,
                            sin_yaw,
                            error_base_states_seq[:, 1:],  # cols 19, 23, 24 (z, vx, vy)
                            ))

    # Assuming error_states, states, and actions are already extracted from `data`
    training_data = np.hstack((states, error_states, state_seq, error_states_seq, current_action, action_seq))

    state_dim = states.shape[1]
    action_dim = action_seq.shape[1]      
    error_dim = error_states.shape[1] 

    dataset = SequenceDataset(training_data, seq_len)

    # Split sizes (e.g., 80% train, 20% validation)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # Randomly split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, state_dim, error_dim, action_dim

def GetData(batch, state_dim, error_dim, action_dim):
    
    initial_state = batch[:, 0, :state_dim].clone().detach().requires_grad_()
    initial_state = initial_state.unsqueeze(1)

    initial_error_state = batch[:,0, state_dim:state_dim+error_dim]
    initial_error_state = initial_error_state.unsqueeze(1)

    state_seq = batch [:, :, :state_dim]
    error_state_seq = batch[:, :, state_dim:state_dim+error_dim]
    new_state_seq = batch[:, :, state_dim+error_dim: 2*state_dim+error_dim]
    new_error_state_seq = batch[:, :, 2*state_dim+error_dim:2*state_dim+2*error_dim]

    action_seq = batch[:, :, -action_dim-action_dim:-action_dim]
    new_action_seq = batch[:, :, -action_dim:]


    return initial_state, initial_error_state, \
            state_seq, new_state_seq, \
            error_state_seq, new_error_state_seq, \
            action_seq, new_action_seq



def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def flatten_state(state_dict):
    def flatten_value(v):
        if isinstance(v, (tuple, list, set, np.ndarray)):
            result = []
            for item in v:
                result.extend(flatten_value(item))  # recursive call
            return result
        else:
            return [v]

    flat = []
    for value in state_dict.values():
        flat.extend(flatten_value(value))
    
    return np.array(flat, dtype=np.float32)


def safe_atan2(y, x, eps=1e-6):
    # Prevent (0, 0)
    mask = (x == 0) & (y == 0)
    x = x + mask * eps
    y = y + mask * eps
    return torch.atan2(y, x)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x