import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import copy
from critic import CriticTransformer
from actor import ActorTransformer
import itertools

class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def add(self, s, g, a, r, s2, g2, d):
        self.buffer.append((s, g, a, r, s2, g2, d))

    
    def sample_sequence(self, sequence_length, batch_size):
        sequences = []

        max_start = len(self.buffer) - sequence_length
        assert max_start > 0, "Not enough data in buffer for a full sequence"

        for _ in range(batch_size):
            start_idx = random.randint(0, max_start)
            seq = list(itertools.islice(self.buffer, start_idx, start_idx + sequence_length))
            sequences.append(seq)

        # Reshape and return as torch tensors
        s, g, a, r, s2, g2, d = map(
            lambda x: torch.FloatTensor(np.stack(x)),
            zip(*[zip(*seq) for seq in sequences])  # Transpose list of sequences
        )

        return s, g, a, r, s2, g2, d

    
class TD3Agent:
    def __init__(self, state_dim, error_dim, action_dim, max_action = 1, device = 'cpu',
                 actor_ckpt = None,
                 hidden_dim = 32, num_layers = 2,
                 actor_lr=1e-4, critic_lr=1e-3, tau=0.005, gamma=0.99, noise_std=0.2, policy_delay=2, seq_len =10):
       
        # Initialize actor
        self.actor = ActorTransformer(state_dim = state_dim, error_dim = error_dim, output_dim = action_dim,
                                      hidden_dim = hidden_dim, num_layers = num_layers).to(device)
        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")
        self.target_actor = copy.deepcopy(self.actor)

        #initialize critic
        self.critic1 = CriticTransformer(state_dim = state_dim, error_dim = error_dim, action_dim = action_dim,
                                        hidden_dim = hidden_dim, num_layers = num_layers, seq_len=seq_len).to(device)
                                         
        self.critic2 = CriticTransformer(state_dim = state_dim, error_dim = error_dim, action_dim = action_dim,
                                        hidden_dim = hidden_dim, num_layers = num_layers, seq_len=seq_len).to(device)
        
        # self.critic1 = CriticTransformer(state_dim = state_dim, error_dim = error_dim, action_dim = action_dim,
        #                                 hidden_dim = hidden_dim).to(device)
                                         
        # self.critic2 = CriticTransformer(state_dim = state_dim, error_dim = error_dim, action_dim = action_dim,
        #                                 hidden_dim = hidden_dim).to(device)
        
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, amsgrad = True, weight_decay=1e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr, amsgrad = True, weight_decay=1e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr, amsgrad = True, weight_decay=1e-4)

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

    def select_action(self, state, error, noise=True):
        # Select action using the actor network (add exploration noise)
        state = state.to(self.device).float()
        error = error.to(self.device).float()
        action = self.actor(state, error)

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


    def train(self, batch_size = 64, noise_clip = 0.5, policy_noise = 0.2, sequence_len = 20):
        batch = self.replay_buffer.sample_sequence(sequence_len, batch_size)
        state, error, action, reward, next_state, next_error, done = batch

       
        state = state.squeeze(2)  # Remove the unnecessary dimension
        error = error.squeeze(2)  # Remove the unnecessary dimension
        action = action.squeeze(2)

        next_state = next_state.squeeze(2)  # Remove the unnecessary dimension
        next_error = next_error.squeeze(2)  # Remove the unnecessary dimension

        state = state.to(self.device).float()
        error = error.to(self.device).float()
        action = action.to(self.device).float()
        reward = reward.to(self.device).float()
        next_state = next_state.to(self.device).float()
        next_error = next_error.to(self.device).float()
        done = done.to(self.device).float()

        state_last = state[:,-1,:]
        error_last = error[:,-1,:]
        next_state_last = next_state[:,-1,:]
        next_error_last = next_error[:,-1,:]
        last_reward = reward[:, -1]        
        last_action = action[:,-1,:]

        # print(f"State shape: {state.shape}")
        # print(f"Error shape: {error.shape}")
        # print(f"Action shape: {action.shape}")
        # print(f"Reward shape: {reward.shape}")
        # print(f"Next state shape: {next_state.shape}")
        # print(f"Next error shape: {next_error.shape}")
        # print(f"Done shape: {done.shape}")
        
        # Get target Q-values from target critics
        with torch.no_grad():
            next_action = self.target_actor(next_state, next_error)
            # print(f"next action shape: {next_action.shape}")
            # Add clipped noise
            noise = torch.normal(0, policy_noise, size=next_action.shape).to(next_action.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q values
            next_action = next_action.to(self.device).float()
            full_action_seq = action.clone()  # [64, n, 4] (copy of the original action sequence)
            full_action_seq[:, -1, :] = next_action  # [64, n, 4]

            target_q1 = self.target_critic1(next_state, next_error, full_action_seq)
            target_q2 = self.target_critic2(next_state, next_error, full_action_seq)
            
            #take the last reward
            
            last_reward = last_reward.unsqueeze(1)  # [64, 1] (adds an extra dimension for consistency)
            # print(f"last reward shape: {last_reward.shape}")

            # Compute target Q values
            target_q = last_reward + self.gamma * torch.min(target_q1, target_q2)

        # Critic loss (MSE between predicted Q-values and target Q-values)
        #take the last action from action sequence
        q1 = self.critic1(state, error, action)
        q2 = self.critic2(state, error, action)

        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        print(f"Average Q1: {q1.mean().item():.4f}",
            f"Average Q2: {q2.mean().item():.4f}",
            f"Average TQ: {target_q.mean().item():.4f}",
            f"reward: {reward.mean().item():.4f}")
        
        # Backpropagate and update critic networks
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss: maximize the Q-value predicted by the critics (using the actor's action)
        # action = self.actor(state, error)
        # q1 = self.critic1(state, error, action)
        # q2 = self.critic2(state, error, action)

        # Actor loss: minimize the negative Q-value (maximize Q-value)
        actor_loss = torch.tensor(0.0)  # Default value if not updated
        self.total_it += 1

        if self.total_it % self.policy_delay == 0:
                # Actor loss (maximize Q from critic1)
                actor_action = self.actor(state, error)
                full_action_seq[:, -1, :] = actor_action  # [64, n, 4]
                actor_loss = -self.critic1(state, error, full_action_seq).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()

    
    def save_model(self):
        """
        Save the actor and critic models to disk.
        """
        # Save the actor model
        torch.save(self.actor.state_dict(), "model/actor.pth")

        # Save the critic models
        torch.save(self.critic1.state_dict(), "model/critic1.pth")
        torch.save(self.critic2.state_dict(), "model/critic2.pth")

        # Optionally, save the target models as well
        torch.save(self.target_actor.state_dict(), "model/actor_target.pth")
        torch.save(self.target_critic1.state_dict(), "model/critic1_target.pth")
        torch.save(self.target_critic2.state_dict(), "model/critic2_target.pth")


