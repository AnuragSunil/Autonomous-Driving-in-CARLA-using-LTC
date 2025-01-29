import torch
import numpy as np
from encoder_init import EncodeState
from networks.off_policy.ddqn.dueling_dqn import DuelingDQnetwork
from networks.off_policy.PER_buffer import PrioritizedReplayBuffer
from parameters import *

class DQNAgent(object):

    def __init__(self, n_actions):
        self.gamma = GAMMA
        self.alpha = DQN_LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.epsilon_decay_rate = EPSILON_DECREMENT  # Slow down decay
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_SIZE, PER_BUFFER_ALPHA, PER_BUFFER_GAMMA)
        self.q_network_eval = DuelingDQnetwork(n_actions, MODEL_ONLINE)
        self.q_network_target = DuelingDQnetwork(n_actions, MODEL_TARGET)

        # Optimizer setup
        self.optimizer = torch.optim.Adam(self.q_network_eval.parameters(), lr=self.alpha)

    def save_transition(self, observation, action, reward, new_observation, done):
        self.replay_buffer.save_transition(observation, action, reward, new_observation, done)

    def get_action(self, observation):
        """ Select action using epsilon-greedy approach """
        if np.random.random() > self.epsilon:
            V, A = self.q_network_eval.forward(observation)
            action = torch.argmax(A).item()  # Select action based on Advantage
        else:
            action = np.random.choice(self.action_space)  # Explore (random action)
        return action

    def decrease_epsilon(self):
        """ Gradual epsilon decay (using exponential decay) at the end of an episode """
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay_rate  # Slower decay (adjusted rate)
        else:
            self.epsilon = self.epsilon_end
        print(f"New epsilon value: {self.epsilon:.5f}")

    def save_model(self):
        """ Save the current model (both evaluation and target networks) """
        self.q_network_eval.save_checkpoint()
        self.q_network_target.save_checkpoint()

    def load_model(self):
        """ Load previously saved models (both evaluation and target networks) """
        self.q_network_eval.load_checkpoint()
        self.q_network_target.load_checkpoint()

    def learn(self):
        """ Learn from experience using Prioritized Experience Replay """
        if len(self.replay_buffer) < self.batch_size:
            return

        self.q_network_eval.optimizer.zero_grad()

        # Update target network periodically with soft updates
        if self.train_step % REPLACE_NETWORK == 0:
            tau = 0.001  # Soft update coefficient
            for target_param, eval_param in zip(self.q_network_target.parameters(), self.q_network_eval.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

        # Sample from replay buffer
        observation, action, reward, new_observation, done, weights, indices = self.replay_buffer.sample_buffer(self.batch_size)

        observation = observation.to(self.q_network_eval.device)
        action = action.to(self.q_network_eval.device)
        reward = reward.to(self.q_network_eval.device)
        new_observation = new_observation.to(self.q_network_eval.device)
        done = done.to(self.q_network_eval.device)
        weights = weights.to(self.q_network_eval.device)

        # Normalize rewards (Reward normalization)
        reward = (reward - reward.mean()) / (reward.std() + 1e-5)

        # Compute predicted V and A for Q-values
        Vs, As = self.q_network_eval.forward(observation)
        nVs, nAs = self.q_network_target.forward(new_observation)

        # Compute Q predictions (using V + A - A.mean()) for evaluation and learning
        q_pred = torch.add(Vs, (As - As.mean(dim=1, keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_next = torch.add(nVs, (nAs - nAs.mean(dim=1, keepdim=True)))
        q_next[done] = 0.0  # If done, the Q value should be 0 for that state

        # Compute target Q-values
        q_target = reward + self.gamma * torch.max(q_next, dim=1)[0].detach()

        # Compute loss using PER weights
        loss = (weights * (q_target - q_pred).pow(2)).mean().to(self.q_network_eval.device)

        # Backpropagate the loss and update model parameters
        loss.backward()

        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network_eval.parameters(), max_norm=1.0)

        self.q_network_eval.optimizer.step()

        # Update priorities in the replay buffer based on TD-error
        td_errors = (q_target - q_pred).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        self.train_step += 1

        # Epsilon decay happens only once at the end of each episode, not per training step
        if done.all():
            self.decrease_epsilon()  # Gradually decrease epsilon at the end of an episode
