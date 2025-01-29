import os
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR, TOWN7

class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DuelingDQnetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR + '/' + TOWN7, model)

        # Deep and wide architecture to improve model capacity
        self.Linear1 = nn.Sequential(
            nn.Linear(128 + 5, 512),  # Increased number of neurons
            nn.LeakyReLU(negative_slope=0.01),  # Use LeakyReLU instead of ReLU
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 256),  # Increased number of neurons
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 128),  # Increased number of neurons
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Separate branches for Value (V) and Advantage (A)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, self.n_actions)

        # Adam optimizer with learning rate defined in parameters
        self.optimizer = optim.AdamW(self.parameters(), lr=DQN_LEARNING_RATE)
        self.loss = nn.MSELoss()

        # Move model to GPU or CPU based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Forward pass through the network
        fc = self.Linear1(x)
        V = self.V(fc)
        A = self.A(fc)

        return V, A  # Always return V and A separately

    def save_checkpoint(self):
        """Save the current model weights."""
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load the model weights from checkpoint."""
        self.load_state_dict(torch.load(self.checkpoint_file))

    def update_target_network(self, target_network):
        """Updates the target network to be the same as the online network."""
        target_network.load_state_dict(self.state_dict())
