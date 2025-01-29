import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from ncps.torch import LTC  # Assuming LTC is defined elsewhere and imported
import logging

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        try:
            # Ensure the device is chosen based on availability
            self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

            # Define the actor and critic LSTMs (LTCs)
            self.actor_ltc = LTC(units=64, input_size=obs_dim).to(self.device)
            self.critic_ltc = LTC(units=64, input_size=obs_dim).to(self.device)

            # Define the fully connected layers for the actor and critic
            self.actor_fc = nn.Linear(64, action_dim).to(self.device)
            self.critic_fc = nn.Linear(64, 1).to(self.device)

            # Define the action standard deviation parameter
            self.action_std = nn.Parameter(torch.full((action_dim,), action_std_init).to(self.device))

        except Exception as e:
            logging.error(f"Error during ActorCritic initialization: {e}")
            raise

    def forward(self, state):
        try:
            state = state.to(self.device)  # Ensure the state is on the correct device
            # Ensure the state has the correct shape (batch size, state_dim)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # Add batch dimension if missing

            # Pass through actor and critic networks
            actor_output, actor_hx = self.actor_ltc(state)
            critic_output, critic_hx = self.critic_ltc(state)

            # Process actor output and critic output
            actor_output = actor_output[:, -1, :] if len(actor_output.shape) == 3 else actor_output.squeeze(1)
            critic_output = critic_output[:, -1, :] if len(critic_output.shape) == 3 else critic_output.squeeze(1)

            # Get the action mean and state value
            action_mean = self.actor_fc(actor_output)
            state_value = self.critic_fc(critic_output).squeeze(0)  # Remove batch dimension

            return action_mean, state_value
        except Exception as e:
            logging.error(f"Error during forward pass: {e}")
            raise


    def get_action_and_log_prob(self, state):
        try:
            action_mean, _ = self.forward(state)

            # Create the distribution and sample an action
            dist = MultivariateNormal(action_mean, torch.diag(self.action_std))  # Ensure action_std is shaped correctly
            action = dist.sample()  # Sample from the distribution
            log_prob = dist.log_prob(action)

            return action, log_prob

        except Exception as e:
            logging.error(f"Error during action selection: {e}")
            raise


    def get_value(self, state):
        try:
            _, state_value = self.forward(state)
            return state_value

        except Exception as e:
            logging.error(f"Error during value computation: {e}")
            raise

    def evaluate(self, states, actions):
        """Evaluate actions and compute log probabilities, state values, and entropy."""
        try:
            # Forward pass through the actor-critic network
            action_mean, values = self.forward(states)

            # Create the distribution
            dist = MultivariateNormal(action_mean, torch.diag(self.action_std).unsqueeze(0))

            # Ensure actions have the correct shape
            actions = actions.view(-1, dist.mean.size(-1))  # Match action dimensions with distribution

            # Debug shapes
            print(f"states shape: {states.shape}")
            print(f"actions shape: {actions.shape}")
            print(f"dist mean shape: {dist.mean.shape}")

            # Compute log probabilities of the actions
            log_probs = dist.log_prob(actions).unsqueeze(-1)  # Add dimension to match others

            # Compute distribution entropy (encourages exploration)
            dist_entropy = dist.entropy().mean()

            return log_probs, values, dist_entropy
        except Exception as e:
            logging.error(f"Error during action evaluation: {e}")
            raise


import torch
from ncps.torch import LTC

# Parameters
input_size = 20  # Input feature size (e.g., state dimension)
units = 50  # Number of hidden units (state size)
batch_size = 2  # Number of samples in a batch
seq_len = 3  # Sequence length (number of time steps)

# Initialize the LTC layer
rnn = LTC(input_size=input_size, units=units)

# Generate random input tensor of shape (batch_size, seq_len, input_size)
x = torch.randn(batch_size, seq_len, input_size)

# Initial hidden state (size = (batch_size, units))
h0 = torch.zeros(batch_size, units)

# Forward pass through the LTC
output, hn = rnn(x, h0)

# Print the output shape and the hidden state
print(f"Output shape: {output.shape}")
print(f"Hidden state shape: {hn[0].shape} (and {hn[1].shape} if mixed memory is used)")
