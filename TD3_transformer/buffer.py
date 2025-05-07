import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, error_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.error_dim = error_dim
        self.action_dim = action_dim

        # The buffer will store state, error, action, reward, next_state, next_error, and done
        self.buffer = deque(maxlen=buffer_size)
        
    def push(self, state, error, action, reward, next_state, next_error, done):
        # Store the experience in the buffer
        self.buffer.append((state, error, action, reward, next_state, next_error, done))
        
    def sample(self, batch_size, exclude_end = 0):

        valid_indices = np.arange(len(self.buffer) - exclude_end) if exclude_end > 0 else np.arange(len(self.buffer))

        # Ensure that we don't try to sample more than the available number of experiences
        sample_size = min(len(valid_indices), batch_size)
        
        # Randomly sample 'batch_size' indices from the valid range
        batch = np.random.choice(valid_indices, sample_size, replace=False)

        # Unzip the batch into individual arrays
        states, errors, actions, rewards, next_states, next_errors, dones = zip(*[self.buffer[i] for i in batch])

        # Convert to numpy arrays or tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        errors = torch.tensor(np.array(errors), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        next_errors = torch.tensor(np.array(next_errors), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, errors, actions, rewards, next_states, next_errors, dones

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
