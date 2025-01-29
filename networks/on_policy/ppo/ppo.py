import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from ncps.torch import LTC
from ncps.wirings import FullyConnected

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim  # Observation dimension (e.g., 133)
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize covariance matrix
        self.cov_var = torch.full((self.action_dim,), action_std_init, device=self.device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        # Define LTC actor network
        wiring = FullyConnected(100)  # Fully connected wiring with 100 units
        self.actor = LTC(
            input_size=self.obs_dim,
            units=wiring,
            return_sequences=False,
            batch_first=True
        )
        self.actor_output_layer = nn.Linear(100, self.action_dim)  # Output layer for action mean

        # Critic network for state-value estimation
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 300),
            nn.Tanh(),
            nn.Linear(300, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        """Update action standard deviation."""
        self.cov_var = torch.full((self.action_dim,), new_action_std, device=self.device)

    def get_value(self, obs):
        """Get state value for a given observation."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.critic(obs)

    def get_action_and_log_prob(self, obs):
        """Get action and its log probability given an observation."""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(1)  # Add batch and time dimensions
        elif obs.ndim == 2:
            obs = obs.unsqueeze(1)  # Add time dimension

        # Forward pass through LTC actor
        h0 = torch.zeros((obs.size(0), 100), device=self.device)  # Initial hidden state for 100 units
        ltc_output, _ = self.actor(obs, h0)
        mean = self.actor_output_layer(ltc_output)

        # Clamp mean to prevent invalid values
        mean = torch.clamp(mean, min=-1e3, max=1e3)

        # Create distribution for action sampling
        dist = MultivariateNormal(mean, self.cov_mat.to(self.device))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

    def evaluate(self, states, actions):
        """Evaluate actions for given states."""
        states = states.to(self.device)
        actions = actions.to(self.device)

        # Forward pass through LTC actor
        if states.ndim == 2:
            states = states.unsqueeze(1)  # Add time dimension
        h0 = torch.zeros((states.size(0), 100), device=self.device)  # Initial hidden state for 100 units
        ltc_output, _ = self.actor(states, h0)
        mean = self.actor_output_layer(ltc_output)

        # Clamp mean to avoid invalid values
        mean = torch.clamp(mean, min=-1e3, max=1e3)

        # Ensure covariance matrix matches expected dimensions
        if self.cov_mat.size(0) != mean.size(0):
            self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0).expand(mean.size(0), -1, -1)

        dist = MultivariateNormal(mean, self.cov_mat.to(self.device))
        logprobs = dist.log_prob(actions)

        # Evaluate state values using critic
        values = self.critic(states.squeeze(1))  # Remove time dimension for critic

        # Calculate distribution entropy for regularization
        dist_entropy = dist.entropy().mean()
        return logprobs.detach(), values.detach(), dist_entropy.detach()
