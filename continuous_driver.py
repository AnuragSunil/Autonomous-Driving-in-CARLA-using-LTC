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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import *
import traceback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE_ACTOR, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by default')
    args = parser.parse_args()
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def runner():
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    if exp_name != 'ppo':
        sys.exit()

    if train:
        writer = SummaryWriter(f"runs/{exp_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{exp_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    action_std_decay_rate = 0.05
    min_action_std = 0.05
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    try:
        client, world = ClientConnection(town).setup()
    except Exception as e:
        traceback.print_exc()
        return

    try:
        env = CarlaEnvironment(client, world, town) if train else CarlaEnvironment(client, world, town, checkpoint_frequency=None)
    except Exception as e:
        traceback.print_exc()
        return

    encode = EncodeState(LATENT_DIM)

    try:
        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_{chkt_file_nums}.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        else:
            agent = PPOAgent(town, action_std_init)
            if not train:
                agent.load()
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False

        if train:
            while timestep < total_timesteps:
                try:
                    observation = env.reset()
                    observation = encode.process(observation)

                    current_ep_reward = 0
                    t1 = datetime.now()

                    for t in range(args.episode_length):
                        action = agent.get_action(observation, train=True)
                        observation, reward, done, info = env.step(action)
                        if observation is None:
                            break
                        observation = encode.process(observation)

                        agent.memory.rewards.append(reward)
                        agent.memory.dones.append(done)
                        timestep += 1
                        current_ep_reward += reward

                        if timestep % action_std_decay_freq == 0:
                            action_std_init = agent.decay_action_std(action_std_decay_rate, min_action_std)

                        if timestep == total_timesteps - 1:
                            agent.chkpt_save()

                        if done:
                            episode += 1
                            t2 = datetime.now()
                            episodic_length.append(abs((t2 - t1).total_seconds()))
                            break

                    deviation_from_center += info[1]
                    distance_covered += info[0]
                    scores.append(current_ep_reward)

                    if checkpoint_load:
                        cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / episode
                    else:
                        cumulative_score = np.mean(scores)

                    print(f"Episode: {episode}, Timestep: {timestep}, Reward: {current_ep_reward:.2f}, Avg Reward: {cumulative_score:.2f}")
                except Exception as e:
                    traceback.print_exc()
        else:
            while timestep < args.test_timesteps:
                pass

    except Exception as e:
        traceback.print_exc()

    finally:
        sys.exit()

if __name__ == "__main__":
    try:
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
