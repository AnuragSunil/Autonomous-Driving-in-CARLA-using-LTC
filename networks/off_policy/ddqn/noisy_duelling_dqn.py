import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

STATE_SIZE = 95

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Noise parameters (mu and sigma)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * std_init)
        self.weight_sigma = nn.Parameter(torch.ones(out_features, in_features) * std_init)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.ones(out_features) * std_init)

    def reset_noise(self):
        # Reset noise (this could involve re-sampling or resetting the noise parameters)
        self.weight_sigma.data.fill_(1.0)  # Reset the standard deviation of the weight noise
        self.bias_sigma.data.fill_(1.0)    # Reset the standard deviation of the bias noise

    def forward(self, input):
        noise_weight = self.weight_sigma * torch.randn_like(self.weight_mu)
        noise_bias = self.bias_sigma * torch.randn_like(self.bias_mu)
        weight = self.weight_mu + noise_weight
        bias = self.bias_mu + noise_bias
        return F.linear(input, weight, bias)

class NoisyDuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(NoisyDuelingDQnetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = model
        
        # Define network layers
        self.fc1 = nn.Linear(95 + 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.V = NoisyLinear(64, 1)
        self.A = NoisyLinear(64, self.n_actions)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move model to the device

    def forward(self, x):
        x = x.to(self.device)  # Ensure the input is moved to the correct device
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        V = self.V(x)
        A = self.A(x)
        return V, A

    def reset_noise(self):
        # Reset noise for both V and A layers
        self.V.reset_noise()
        self.A.reset_noise()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))