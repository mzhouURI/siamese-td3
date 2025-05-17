import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import copy
from network.modeler_transformer import VehicleModeler
from network.actor_transformer import VehicleActor
import itertools
from network.utilites import safe_atan2


class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, s, ns, e, ne, a):
        self.buffer.append((s, ns, e, ne, a))

    
    def sample_sequence(self, sequence_length, batch_size):
        sequences = []

        max_start = len(self.buffer) - sequence_length
        assert max_start > 0, "Not enough data in buffer for a full sequence"

        for _ in range(batch_size):
            start_idx = random.randint(0, max_start)
            seq = list(itertools.islice(self.buffer, start_idx, start_idx + sequence_length))
            sequences.append(seq)
    
        # Reshape and return as torch tensors
        s, ns, e, ne, a = map(
            lambda x: torch.FloatTensor(np.stack(x)),
            zip(*[zip(*seq) for seq in sequences])  # Transpose list of sequences
        )
        return s, ns, e, ne, a

class RL_MPC_Agent:
    def __init__(self, state_dim, error_dim, action_dim, max_action = 1, device = 'cpu',
                 actor_ckpt = None, modeler_ckpt = None,
                 hidden_dim = 32, num_layers = 2, num_head =4,
                 actor_lr=1e-4, modeler_lr=1e-3, 
                 tau=0.005, gamma=0.99, noise_std=0.2, policy_delay=2,
                 pitch_loss_weight = 1, depth_loss_weight =1, surge_loss_weight =1, yaw_loss_weight =1,
                 smooth_loss_weight =0.1, jerk_loss_weight = 0.1):
        
        self.pitch_weight = pitch_loss_weight
        self.depth_weight = depth_loss_weight
        self.surge_weight = surge_loss_weight
        self.yaw_weight = yaw_loss_weight
        self.smooth_weight = smooth_loss_weight
        self.jerk_weight = jerk_loss_weight
        # Initialize actor
        self.actor = VehicleActor(state_dim = state_dim+error_dim, action_dim = action_dim,
                                      d_model = hidden_dim, nhead = num_head,
                                      num_layers = num_layers, max_action = max_action).to(device)
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")
        self.target_actor = copy.deepcopy(self.actor)

        #initialize critic
        self.modeler = VehicleModeler(state_dim = state_dim, action_dim = action_dim,
                                        d_model = hidden_dim, nhead = num_head, 
                                        num_layers = num_layers).to(device)                           
        
        if modeler_ckpt is not None:
            self.modeler.load_state_dict(torch.load(modeler_ckpt, map_location=device))
            print(f"Loaded actor weights from {modeler_ckpt}")
        self.target_modeler = copy.deepcopy(self.modeler)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad = True, weight_decay=1e-4)
        self.modeler_optimizer = optim.Adam(self.modeler.parameters(), lr=modeler_lr, amsgrad = True, weight_decay=1e-4)
        
        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.noise_std = noise_std
        self.policy_delay = policy_delay
        self.total_timesteps = 0
        self.max_action = max_action
        self.device = device
        self.total_it = 0
        self.replay_buffer = ReplayBuffer()

    
    def select_action(self, state, seq_len, noise=True):
        # Select action using the actor network (add exploration noise)
        state = state.to(self.device).float()
        action_seq = self.actor.forward(state, seq_len)

        if torch.isnan(action_seq).any():
                print("NaN detected in action_seq!")
                exit()
        #only use the first one
        action = action_seq[:,1,:]
        if noise:
            noise_tensor = torch.normal(
                mean=0.0,
                std=self.noise_std,
                size=action.shape,
                device=action.device
            )
            action += noise_tensor
        # Clip actions to valid range
        action = torch.clamp(action, -self.max_action, self.max_action)

        return action
    

    def save_model(self):
        """
        Save the actor and critic models to disk.
        """
        # Save the actor model
        torch.save(self.actor.state_dict(), "model/actor.pth")

        # Save the critic models
        torch.save(self.modeler.state_dict(), "model/modeler.pth")
        # Optionally, save the target models as well
        torch.save(self.target_actor.state_dict(), "model/actor_target.pth")

    def train(self, batch_size = 64, sequence_len = 20):
        batch = self.replay_buffer.sample_sequence(sequence_len, batch_size)
        state_seq, new_state_seq, error_seq, new_error_seq, action_seq = batch

        state_seq = state_seq.to(self.device).float()
        error_seq = error_seq.to(self.device).float()
        action_seq = action_seq.to(self.device).float()
        new_state_seq = new_state_seq.to(self.device).float()
        
        state_seq = state_seq.squeeze(2)
        new_state_seq = new_state_seq.squeeze(2)
        error_seq = error_seq.squeeze(2)
        action_seq = action_seq.squeeze(2)
        
        # Train modeler
  
        #get the first state
        initial_state = state_seq[:,0,:]
        initial_state = initial_state.unsqueeze(1)
        zero_depth_initial_state = initial_state.clone()

        #zero depth for reference
        zero_depth_initial_state[:,:,0] = 0
        
        #predict two sequence of future states
        model_pred = self.modeler.forward(zero_depth_initial_state, action_seq)
        # model_pred = self.target_modeler.forward(zero_depth_initial_state, action_seq)

        #add depth back
        model_pred [:,:,0] = model_pred[:,:,0] + initial_state[:,:,0]

        modeler_loss = nn.MSELoss(reduction= "sum")(model_pred, new_state_seq)
        
        # Backpropagate and update critic networks
        self.modeler_optimizer.zero_grad()
        modeler_loss.backward()
        self.modeler_optimizer.step()


        # Actor loss: minimize the negative Q-value (maximize Q-value)
        actor_loss = torch.tensor(0.0)  # Default value if not updated
        self.total_it += 1

        if self.total_it % self.policy_delay == 0:
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
            # Actor loss (maximize Q from critic1)
            #only update using the modeler output
            ##get the desired state
            c_depth = error_seq[:,:,ind_e_z] + state_seq[:,:,ind_s_z]
            c_u = error_seq[:,:,ind_e_u] + state_seq[:, : ,ind_s_u]

            e_cos_pitch = error_seq[:, :, ind_e_cos_pitch]
            e_sin_pitch = error_seq[:, :, ind_e_sin_pitch]
            e_cos_yaw = error_seq[:, :, ind_e_cos_yaw]
            e_sin_yaw = error_seq[:, :, ind_e_sin_yaw]

            m_cos_pitch = state_seq[:, :, ind_s_cos_pitch]
            m_sin_pitch = state_seq[:, :, ind_s_sin_pitch]
            m_cos_yaw = state_seq[:, :, ind_s_cos_yaw]
            m_sin_yaw = state_seq[:, :, ind_s_sin_yaw]

            c_sin_pitch = e_sin_pitch*m_cos_pitch + e_cos_pitch*m_sin_pitch
            c_cos_pitch = e_cos_pitch*m_cos_pitch - e_sin_pitch*m_sin_pitch

            c_sin_yaw = e_sin_yaw*m_cos_yaw + e_cos_yaw*m_sin_yaw
            c_cos_yaw = e_cos_yaw*m_cos_yaw - e_sin_yaw*m_sin_yaw

            initial_error_state = error_seq[:,0,:] 
            initial_error_state = initial_error_state.unsqueeze(1)
            
            actor_states= torch.cat([zero_depth_initial_state, initial_error_state], dim = 2)

            pred_action_seq= self.actor.forward(actor_states, sequence_len) 
            # pred_action_seq= self.target_actor.forward(actor_states, sequence_len) 

            #using model
            # pred_states = self.modeler.forward(zero_depth_initial_state, pred_action_seq)
            pred_states = self.target_modeler.forward(zero_depth_initial_state, pred_action_seq)


            #add initial depth back
            pred_states [:,:,0] = pred_states[:,:,0] + initial_state[:,:,0]

            #compute difference
            p_depth = pred_states[:,:,ind_s_z]
            p_u = pred_states[:, : ,ind_s_u]

            p_cos_pitch = pred_states[:, :, ind_s_cos_pitch]
            p_sin_pitch = pred_states[:, :, ind_s_sin_pitch]

            p_cos_yaw = pred_states[:, :, ind_s_cos_yaw]
            p_sin_yaw = pred_states[:, :, ind_s_sin_yaw]
            
            #desired - predition
            pred_e_cos_pitch = c_cos_pitch*p_cos_pitch + c_sin_pitch *p_sin_pitch
            pred_e_sin_pitch = c_sin_pitch*p_cos_pitch - c_cos_pitch*p_sin_pitch
            pred_e_cos_yaw = c_cos_yaw*p_cos_yaw + c_sin_yaw *p_sin_yaw
            pred_e_sin_yaw = c_sin_yaw*p_cos_yaw - c_cos_yaw*p_sin_yaw


            pred_e_depth = c_depth - p_depth
            pred_e_u     = c_u - p_u
            pred_e_pitch = safe_atan2(pred_e_sin_pitch, pred_e_cos_pitch)
            
            pred_e_yaw = safe_atan2(pred_e_sin_yaw, pred_e_cos_yaw)

        
            #error between desired and the predicted states
            # pred_e_pitch = 1*pred_e_pitch
            # pred_e_u = 5*pred_e_u
            pred_e = torch.stack([self.depth_weight*pred_e_depth, 
                                  self.pitch_weight*pred_e_pitch, 
                                  self.yaw_weight*pred_e_yaw, 
                                  self.surge_weight*pred_e_u ], dim = -1)
                
            actor_loss = torch.mean(pred_e**2) 

            delta_action = pred_action_seq[:,1:,:] - pred_action_seq[:,:-1,:]
            action_smooth_loss = torch.mean(delta_action **2)


            jerk = pred_action_seq[:,2:,:] - 2* pred_action_seq[:,1:-1,:] + pred_action_seq[:,:-2,:]
            jerk_loss = torch.mean(jerk **2)   

            energy_loss = torch.mean(abs(pred_action_seq)**2)
            # print(delta_action.shape)
            # print(action_smooth_loss)

            # print(jerk.shape)
            # print(jerk_loss)

            total_loss = actor_loss + self.smooth_weight*action_smooth_loss + self.jerk_weight * jerk_loss + 0.0 *energy_loss
            #use hybre loss for teacher forcing
            # delta_action = action_seq - pred_action_seq
            # actor_loss = torch.sum(delta_action **2) + torch.sum(new_error_seq**2)
            # print(actor_loss.item())
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

             # Soft update target networks
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

        for target_param, param in zip(self.target_modeler.parameters(), self.modeler.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return modeler_loss.item(), actor_loss.item()