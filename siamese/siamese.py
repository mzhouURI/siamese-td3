import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
    
class PoseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(PoseEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)


    def forward(self, pose):
        x = F.relu(self.fc1(pose))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.manual_seed(42)  # Fix seed per layer for repeatability
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class SiamesePoseControlNet(nn.Module):
    def __init__(self, current_pose_dim=3, goal_pose_dim=2, latent_dim=32, thruster_num =4):
        super(SiamesePoseControlNet, self).__init__()
        self.current_encoder = PoseEncoder(input_dim=current_pose_dim, latent_dim=latent_dim)
        self.current_encoder.apply(self.current_encoder.init_weights)

        self.goal_encoder = PoseEncoder(input_dim=goal_pose_dim, latent_dim=latent_dim)
        self.goal_encoder.apply(self.current_encoder.init_weights)

        self.control_head = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, thruster_num)  # Output: [v_linear, v_angular, etc.]
            # nn.Tanh(),
        )

    def forward(self, current_pose, goal_pose):
        current_feat = self.current_encoder(current_pose)
        goal_feat = self.goal_encoder(goal_pose)
        combined = torch.cat((current_feat, goal_feat), dim=1)
        control = self.control_head(combined)
        return control
    
    
class OnlineTrainer:
    def __init__(self, model, lr = 2e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction = 'sum')

    def train(self, predicted_control, true_control, error_pose):
        # Ensure both inputs are torch tensors and of correct type
        if isinstance(predicted_control, list):
            predicted_control = torch.tensor(predicted_control, dtype=torch.float32)
        if isinstance(true_control, list):
            true_control = torch.tensor(true_control, dtype=torch.float32)

        # Ensure inputs are 2D (batch_size, control_dim), if they are not already
        if predicted_control.ndimension() == 1:  # If it's 1D (e.g., single sample)
            predicted_control = predicted_control.unsqueeze(0)

        if true_control.ndimension() == 1:  # If it's 1D (e.g., single sample)
            true_control = true_control.unsqueeze(0)

        if error_pose.ndimension() == 1:
            error_pose = error_pose.unsqueeze(0)

        # Make sure the dimensions of predicted and true control match
        assert predicted_control.shape == true_control.shape, \
            f"Shape mismatch: predicted {predicted_control.shape}, true {true_control.shape}"
        error_pose = np.array(error_pose)
        state_loss = np.linalg.norm(error_pose)

        # Compute MSE loss between the batch of predicted and true control commands
        loss = self.loss_fn(predicted_control, true_control) + 0.0 * state_loss

        # Perform the backward pass and optimize
        self.optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagate gradients

        # Compute the norm of gradients
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad, p=2)  # L2 norm of each gradient tensor
                total_norm += grad_norm.item() ** 2  # Sum squared norms

        total_norm = total_norm ** 0.5  # Take the square root to get the total norm
        # print(f"Gradient norm: {total_norm}")
        self.optimizer.step()  # Perform an optimization step

        return loss.item()

    def save_model(self):
        torch.save(self.model.state_dict(), "siamese_pose_control_net.pth")
        