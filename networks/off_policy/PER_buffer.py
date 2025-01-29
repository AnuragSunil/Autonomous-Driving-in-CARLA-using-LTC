import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, mem_size, alpha, beta):
        self.mem_size = mem_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 0.001
        self.buffer = []
        self.priorities = np.zeros((mem_size,), dtype=np.float32)
        self.max_priority = 1.0
        self.position = 0

    def save_transition(self, observation, action, reward, new_observation, done):
        max_priority = self.max_priority if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.mem_size:
            self.buffer.append((observation, action, reward, new_observation, done))
        else:
            self.buffer[self.position] = (observation, action, reward, new_observation, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.mem_size

    def sample_buffer(self, batch_size):
        # Ensure no NaN values in priorities
        if len(self.buffer) == self.mem_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # Prevent NaN values in priorities
        priorities = np.nan_to_num(priorities, nan=1e-6)

        # Uniform sampling fallback if all priorities are zero
        if priorities.sum() == 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        weights /= weights.max()

        observations, actions, rewards, new_observations, dones = zip(*samples)

        # Convert data to numpy or tensor format
        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return np.array(data, dtype=np.float32)

        observations = np.array([to_numpy(obs) for obs in observations], dtype=np.float32)
        actions = np.array([to_numpy(act) for act in actions], dtype=np.int64)
        rewards = np.array([to_numpy(reward) for reward in rewards], dtype=np.float32)
        new_observations = np.array([to_numpy(new_obs) for new_obs in new_observations], dtype=np.float32)
        dones = np.array([to_numpy(done) for done in dones], dtype=bool)

        return (torch.tensor(observations, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(new_observations, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool),
                torch.tensor(weights, dtype=torch.float32),
                indices)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = np.abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
