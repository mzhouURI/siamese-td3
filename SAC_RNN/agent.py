import torch
import torch.nn.functional as F
import copy
from networks import RNNCritic, RNNActor, RNNReplayBuffer


import torch
import random
import numpy as np
from collections import deque

class SACAgent:
    def __init__(self,
        obs_dim, action_dim, hidden_size=256, rnn_layer = 1, seq_len = 10,
        actor_ckpt = None, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
        tau=0.005, gamma=0.99, target_entropy=None, device='cpu'):

        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = RNNActor(obs_dim, action_dim, hidden_size, rnn_layer).to(device)

        if actor_ckpt is not None:
            self.actor.load_state_dict(torch.load(actor_ckpt, map_location=device))
            print(f"Loaded actor weights from {actor_ckpt}")

        self.critic1 = RNNCritic(obs_dim, action_dim, hidden_size, rnn_layer).to(device)
        self.critic2 = RNNCritic(obs_dim, action_dim, hidden_size, rnn_layer).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        #replay buffer
        self.replay_buffer = RNNReplayBuffer(1000000, seq_len, obs_dim, action_dim, device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy temperature
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -action_dim if target_entropy is None else target_entropy

        self.alpha = self.log_alpha.exp().detach()

        # Sync targets
        self._soft_update(self.critic1, self.target_critic1, tau=1.0)
        self._soft_update(self.critic2, self.target_critic2, tau=1.0)

    def select_action(self, obs_seq, hidden_state=None, deterministic=False):
        self.actor.eval()
        with torch.no_grad():
            obs_seq = obs_seq.to(self.device).unsqueeze(0)  # Add batch dim
            action, _, _ = self.actor(obs_seq, hidden_state, deterministic)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update(self, batch_size):
        obs_seq, action_seq, reward_seq, next_obs_seq, _ = self.replay_buffer.sample(batch_size)
        B, T, _ = obs_seq.shape

        # Move to device
        obs_seq = obs_seq.to(self.device)
        action_seq = action_seq.to(self.device)
        reward_seq = reward_seq.to(self.device)
        next_obs_seq = next_obs_seq.to(self.device)

        # Init hidden states
        h1 = self.critic1.init_hidden(B)
        h2 = self.critic2.init_hidden(B)
        th1 = self.target_critic1.init_hidden(B)
        th2 = self.target_critic2.init_hidden(B)
        ha = self.actor.init_hidden(B)

        with torch.no_grad():
            next_action, _, next_log_prob = self.actor(next_obs_seq, ha)
            target_q1, _ = self.target_critic1(next_obs_seq, next_action, th1)
            target_q2, _ = self.target_critic2(next_obs_seq, next_action, th2)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward_seq + self.gamma * target_q

        # Critic losses
        q1_pred, _ = self.critic1(obs_seq, action_seq, h1)
        q2_pred, _ = self.critic2(obs_seq, action_seq, h2)

        critic1_loss = F.mse_loss(q1_pred, target_value)
        critic2_loss = F.mse_loss(q2_pred, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        new_action, _, log_prob = self.actor(obs_seq, ha, return_log_prob=True)
        q1_pi, _ = self.critic1(obs_seq, new_action, h1)
        q2_pi, _ = self.critic2(obs_seq, new_action, h2)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Target update
        self._soft_update(self.critic1, self.target_critic1, self.tau)
        self._soft_update(self.critic2, self.target_critic2, self.tau)

        return critic1_loss.item(), critic2_loss.item(),actor_loss.item(), alpha_loss.item(), self.alpha.item()
        

    def _soft_update(self, source, target, tau):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

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


