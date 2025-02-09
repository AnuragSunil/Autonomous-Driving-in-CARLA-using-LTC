import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from threading import Thread
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from networks.off_policy.ddqn.agent import DQNAgent
from encoder_init import EncodeState
from parameters import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=DQN_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-episodes', type=int, default=EPISODES, help='total timesteps of the experiment')
    parser.add_argument('--train', type=bool, default=True, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args


def runner():
    try:
        #========================================================================
        #                           BASIC PARAMETER & LOGGING SETUP
        #========================================================================
        args = parse_args()
        exp_name = args.exp_name
        print("Arguments parsed successfully.")

        try:
            if exp_name == 'ddqn':
                run_name = f"DDQN"
        except Exception as e:
            print(f"Error in determining experiment name: {e}")
            sys.exit()

        town = args.town
        writer = SummaryWriter(f"runs/{run_name}/{town}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])),
        )
        
        print(f"Experiment {exp_name} initialized with town {town}.")
        
        # Seeding to reproduce the results 
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        
        #========================================================================
        #                           INITIALIZING THE NETWORK
        #========================================================================
        print("Initializing DQN Agent...")
        checkpoint_load = args.load_checkpoint
        n_actions = 7  # Car can only make 7 actions
        agent = DQNAgent(n_actions)  # Slower epsilon decay
        
        epoch = 0
        cumulative_score = 0
        episodic_length = list()
        scores = list()
        deviation_from_center = 0
        distance_covered = 0

        if checkpoint_load:
            print("Loading checkpoint...")
            agent.load_model()
            if exp_name == 'ddqn':
                with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'rb') as f:
                    data = pickle.load(f)
                    epoch = data['epoch']
                    cumulative_score = data['cumulative_score']
                    agent.epsilon = data['epsilon']
            print("Checkpoint loaded successfully.")

        #========================================================================
        #                           CREATING THE SIMULATION
        #========================================================================
        try:
            print("Setting up connection to CARLA...")
            client, world = ClientConnection(town).setup()
            logging.info("Connection has been set up successfully.")
        except Exception as e:
            logging.error(f"Error while setting up the client connection: {e}")
            sys.exit(1)
        
        print("CARLA connection successful.")
        env = CarlaEnvironment(client, world, town, continuous_action=False)
        encode = EncodeState(LATENT_DIM)

        try:
            time.sleep(1)
            #========================================================================
            #                           INITIALIZING THE MEMORY
            #========================================================================
            if exp_name == 'ddqn' and checkpoint_load:
                while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
                    print(f"Filling replay buffer: {agent.replay_buffer.counter}/{agent.replay_buffer.buffer_size}")
                    observation = env.reset()
                    observation = encode.process(observation)
                    done = False
                    while not done:
                        action = random.randint(0, n_actions - 1)
                        new_observation, reward, done, _ = env.step(action)
                        reward = reward / max(abs(reward), 1.0)  # Normalize reward
                        new_observation = encode.process(new_observation)
                        agent.save_transition(observation, action, reward, new_observation, int(done))
                        observation = new_observation
                    print("Replay buffer filled.")

            #========================================================================
            #                           ALGORITHM
            #========================================================================
            if args.train:
                print("Starting training...")
                for step in range(epoch + 1, EPISODES + 1):
                    print(f"Starting Episode: {step}, Epsilon Now: {agent.epsilon:.3f}")

                    # Reset
                    done = False
                    observation = env.reset()
                    observation = encode.process(observation)
                    current_ep_reward = 0

                    # Episode start: timestamp
                    t1 = datetime.now()

                    while not done:
                        action = agent.get_action(observation)
                        new_observation, reward, done, info = env.step(action)
                        reward = reward / max(abs(reward), 1.0)  # Normalize reward
                        if new_observation is None:
                            print("Received None observation, breaking the loop.")
                            break
                        new_observation = encode.process(new_observation)
                        current_ep_reward += reward

                        agent.save_transition(observation, action, reward, new_observation, int(done))
                        agent.learn()

                        observation = new_observation

                    # Episode end: timestamp
                    t2 = datetime.now()
                    t3 = t2 - t1
                    episodic_length.append(abs(t3.total_seconds()))

                    deviation_from_center += info[1]
                    distance_covered += info[0]

                    scores.append(current_ep_reward)

                    if checkpoint_load:
                        cumulative_score = ((cumulative_score * (step - 1)) + current_ep_reward) / (step)
                    else:
                        cumulative_score = np.mean(scores)
                    
                    agent.decrease_epsilon()
                    print(f"Episode {step} finished. Reward: {current_ep_reward:.2f}, Average Reward: {cumulative_score:.2f}")

                    # Save model periodically
                    if step >= 10 and step % 10 == 0:
                        print(f"Saving model at episode {step}...")
                        agent.save_model()

                        if exp_name == 'ddqn':
                            data_obj = {'cumulative_score': cumulative_score, 'epsilon': agent.epsilon, 'epoch': step}
                            with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'wb') as handle:
                                pickle.dump(data_obj, handle)

                        writer.add_scalar("Cumulative Reward/info", cumulative_score, step)
                        writer.add_scalar("Epsilon/info", agent.epsilon, step)
                        writer.add_scalar("Episodic Reward/episode", scores[-1], step)
                        writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-10]), step)
                        writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), step)
                        writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 10, step)
                        writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 10, step)

                        episodic_length = list()
                        deviation_from_center = 0
                        distance_covered = 0

                print("Training completed.")
                sys.exit()

            else:
                print("Training is disabled.")
                sys.exit()

        except Exception as e:
            print(f"An error occurred during the main loop: {e}")
            logging.error(f"Error in main loop: {str(e)}", exc_info=True)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        sys.exit()

    finally:
        print("\nExit")


if __name__ == "__main__":
    try:    
        runner()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
