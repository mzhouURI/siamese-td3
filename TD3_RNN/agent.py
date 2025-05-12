import torch
import torch.nn.functional as F
import copy
from networks import RNNCritic, RNNActor, RNNReplayBuffer
import torch.nn as nn


import torch
import random
import numpy as np
from collections import deque

class TD3Agent:
    def __init__(self,
        obs_dim, action_dim, hidden_size=256, rnn_layer = 1, seq_len = 10,
        actor_ckpt = None, actor_lr=3e-4, critic_lr=3e-4,
        tau=0.005, gamma=0.99, noise_std = 0.2, policy_delay =2,
        device='cpu',max_action = 1):

        # Hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.noise_std = noise_std
        self.policy_delay = policy_delay
        self.total_timesteps = 0
        self.max_action = max_action
        self.device = device
        self.total_it = 0

        self.actor = RNNActor(obs_dim, action_dim, hidden_size, rnn_layer).to(device)

        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")
        self.target_actor = copy.deepcopy(self.actor)

        self.critic1 = RNNCritic(obs_dim+action_dim, hidden_size, rnn_layer).to(device)
        self.critic2 = RNNCritic(obs_dim+action_dim, hidden_size, rnn_layer).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        #replay buffer
        self.replay_buffer = RNNReplayBuffer(1000000, seq_len, obs_dim, action_dim, device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)



    def select_action(self, obs_seq, noise=True):
        # Select action using the actor network (add exploration noise)
        # obs_seq = obs_seq.to(self.device).float()
        # B, T, _ = obs_seq.shape

        # ha = self.actor.init_hidden(B)
        # obs_seq = obs_seq.to(self.device).float()

        action = self.actor(obs_seq, None)
        # action = action[:, -1, :]  # Shape: (batch_size, state_dim)
        # action = action.detach().squeeze()
        # print(action.shape)
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

    def update(self, batch_size):
        obs_seq, action_seq, reward_seq, next_obs_seq, _ = self.replay_buffer.sample(batch_size)
        B, T, _ = obs_seq.shape

        # Move to device
        obs_seq = obs_seq.to(self.device)
        action_seq = action_seq.to(self.device)
        action_seq = action_seq.squeeze(2)

        reward_seq = reward_seq.to(self.device)
        next_obs_seq = next_obs_seq.to(self.device)

        # Init hidden states
        # h1 = self.critic1.init_hidden(B)
        # h2 = self.critic2.init_hidden(B)
        # th1 = self.target_critic1.init_hidden(B)
        # th2 = self.target_critic2.init_hidden(B)
        # ha = self.actor.init_hidden(B)

        with torch.no_grad():
            next_action = self.actor(next_obs_seq)
            # next_action = n_action[:, -1, :]  # Shape: (batch_size, state_dim)
            # next_action = next_action.detach().squeeze()

            noise = torch.normal(0, self.noise_std, size=next_action.shape).to(self.device)
            noise = noise.clamp(-0.5, 0.5)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # next_action_seq = action_seq.clone()  # [64, n, 4] (copy of the original action sequence)
            # next_action_seq[:,:-1,:] = action_seq[:, 1:, :]
            # next_action_seq[:, -1, :] = next_action  # [64, n, 4]

            next_action[:, :-1, :] = action_seq[:, 1:, :] #replace some next action with exiisting action seq.

            critic_states = torch.cat([next_obs_seq, next_action], dim=-1)

            target_q1 = self.target_critic1(critic_states)
            target_q2 = self.target_critic2(critic_states)

            target_q1 = target_q1[:, -1, :]  # Shape: (batch_size, state_dim)
            # target_q1 = target_q1.detach().squeeze()
            target_q2 = target_q2[:, -1, :]  # Shape: (batch_size, state_dim)
            # target_q2 = target_q2.detach().squeeze()

            reward = reward_seq[:, -1, :]
            target_q = reward + self.gamma * torch.min(target_q1, target_q2)

        # Critic losses
        critic_states = torch.cat([obs_seq, action_seq], dim=-1)
        q1 = self.critic1(critic_states)
        q2 = self.critic2(critic_states)

        q1 = q1[:, -1, :]  # Shape: (batch_size, state_dim)
        q2 = q2[:, -1, :]  # Shape: (batch_size, state_dim)

        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        print(f"Average Q1: {q1.mean().item():.4f}",
            f"Average Q2: {q2.mean().item():.4f}",
            f"Average TQ: {target_q.mean().item():.4f}",
            f"reward: {reward_seq.mean().item():.4f}")

        self.critic1_optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # print("check point")
        self.critic2_optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # print("check point2")

        # Actor loss: minimize the negative Q-value (maximize Q-value)
        actor_loss = torch.tensor(0.0)  # Default value if not updated
        self.total_it += 1

        if self.total_it % self.policy_delay == 0:
            # Actor loss (maximize Q from critic1)

            actor_action = self.actor(obs_seq)
            # action_seq[:, -1] = actor_action[:, -1, :].detach().squeeze()

            actor_action[:, :-1, :] = action_seq[:, 1:, :] #plug existing action seq except the first one into actor

            critic_states = torch.cat([obs_seq, actor_action], dim=-1)

            q = self.critic1(critic_states)

            q = q[:, -1, :]  # Shape: (batch_size, state_dim)

            actor_loss = - q.mean()
            
            self.actor_optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
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
        torch.save(self.target_critic1.state_dict(), "model/critic1_target.pth")
        torch.save(self.target_critic2.state_dict(), "model/critic2_target.pth")


