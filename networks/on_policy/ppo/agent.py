import os
import numpy as np
import torch
import torch.nn as nn
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic  # Importing updated ActorCritic with LTC
from parameters import *

# Ensure that the device is CUDA if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self):
        self.observation = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.observation.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOAgent(object):
    def __init__(self, town, action_std_init=0.4):
        self.obs_dim = 133
        self.action_dim = 2
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = 7
        self.lr_actor = PPO_LEARNING_RATE_ACTOR
        self.lr_critic = PPO_LEARNING_RATE_CRITIC
        self.action_std = action_std_init
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        self.checkpoint_file_no = 0

        # ActorCritic initialization
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
    def get_action(self, obs, train):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

            action, log_prob = self.old_policy.get_action_and_log_prob(obs)

        if train:
            self.memory.observation.append(obs.clone().detach())
            # Ensure actions have consistent shape
            action = action if action.dim() == 2 else action.unsqueeze(0)
            self.memory.actions.append(action)
            self.memory.log_probs.append(log_prob)
            self.memory.rewards.append(0)  # Initialize rewards to 0, as rewards are not provided in the get_action method
            self.memory.dones.append(False)  # Initialize dones to False, as dones are not provided in the get_action method
        else:
            action = action.detach().cpu().numpy().flatten()
        
        return action

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = max(self.action_std - action_std_decay_rate, min_action_std)
        self.set_action_std(self.action_std)
        return self.action_std

    def learn(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards and values
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_states = torch.stack(self.memory.observation).detach().to(self.device)
        values = self.policy.get_value(old_states).squeeze()
        rewards, values = rewards - rewards.mean(), values - values.mean()
        rewards, values = rewards / (rewards.std() + 1e-7), values / (values.std() + 1e-7)

        # Retrieve stored data using the Buffer class
        old_actions = torch.stack([a.unsqueeze(0) if a.dim() == 1 else a for a in self.memory.actions]).detach().to(self.device)
        old_logprobs = torch.tensor(self.memory.log_probs).detach().to(self.device)
        dones = torch.tensor(self.memory.dones).bool().to(self.device)

        # Ensure that the number of observations matches the number of rewards
        num_observations = len(self.memory.observation)
        num_rewards = len(self.memory.rewards)
        if num_observations != num_rewards:
            print(f"Warning: Number of observations ({num_observations}) does not match number of rewards ({num_rewards}). Truncating rewards to match the number of observations.")
            self.memory.rewards = self.memory.rewards[:num_observations]
            self.memory.observation = self.memory.observation[:num_observations]

        # Compute advantages
        advantages = rewards - values.detach()

        # Optimize policy
        for _ in range(self.n_updates_per_iteration):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            assert old_actions.shape == logprobs.shape, f"Action mismatch: {old_actions.shape} vs {logprobs.shape}"

            # PPO ratio
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            # Loss function
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values.squeeze(), rewards) - 0.01 * dist_entropy

            # Update policy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Don't forget to clear the buffer after learning
        self.memory.clear()

    def save(self):
        os.makedirs(PPO_CHECKPOINT_DIR + self.town, exist_ok=True)
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2])
        checkpoint_file = f"{PPO_CHECKPOINT_DIR}{self.town}/ppo_policy_{self.checkpoint_file_no}_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def chkpt_save(self):
        os.makedirs(PPO_CHECKPOINT_DIR + self.town, exist_ok=True)
        self.checkpoint_file_no = max(len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2]) - 1, 0)
        checkpoint_file = f"{PPO_CHECKPOINT_DIR}{self.town}/ppo_policy_{self.checkpoint_file_no}_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)

    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR + self.town))[2]) - 1
        checkpoint_file = f"{PPO_CHECKPOINT_DIR}{self.town}/ppo_policy_{self.checkpoint_file_no}_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file))
        self.policy.load_state_dict(torch.load(checkpoint_file))