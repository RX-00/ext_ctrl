'''

Testing program for pre-trained policy that's supposed to learn the behavior
of a state-space controller (e.g. LQR) on a cartpole in a nominal environment (i.e.
the tilt of the track is zero, the pendulum doesn't have even mass distribution).

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

import numpy as np
import mujoco
import mediapy as media
import matplotlib.pyplot as plt

import os

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs

from ppo import PPO


def test():
    print("Testing pre-trained policy...")
    env_id = "NominalCartpole"
    render_mode = "human"                 # NOTE: depth_array for no render, human for yes render
    ep_len_max = 500                      # max timesteps in one episode
    # NOTE: action_std_dev needs to be the SAME
    #       as the action distrbution which was used while saving
    action_std_dev = 0.1            # initial std dev for action distr (Multivariate Normal, i.e. Gaussian)
    
    num_test_eps = 10

    K_epochs = 80                       # update policy for K epochs in a single PPO update
    eps_clip = 0.2                      # clip param for PPO-Clip Objective Function
    gamma = 0.99                        # discount factor
    lr_actor = 0.0003                   # learning rate for actor NN
    lr_critic = 0.001                   # learning rate for critic NN

    print("Gymnasium env: " + env_id)

    env = gym.make(env_id, render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppoAgent = PPO(state_dim, action_dim, lr_actor, lr_critic,
                    gamma, K_epochs, eps_clip, action_std_dev)
    
    # loading up weights
    # NOTE: Choose the number of the pretrained model weights you want to test
    run_num_pretrained = 0

    directory = "ext_ctrl_pretrained" + '/' + env_id + '/'
    checkpoint_path = directory + "ext_ctrl_{}_{}.pth".format(env_id, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppoAgent.load(checkpoint_path)


    test_running_reward = 1

    for ep in range(1, num_test_eps + 1):
        ep_reward = 0
        state = env.reset()[0]
        env.render()

        for ts in range(1, ep_len_max + 1):
            action = ppoAgent.sel_action(state)
            state, reward, done, _, _ = env.step(action)
            ep_reward += reward

            # if done:
            #     break
        
        # clear rollout buffer
        ppoAgent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        
        # reset episode reward for next episode
        ep_reward
    
    env.close()

    avg_test_reward = test_running_reward / num_test_eps
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    '''
    TODO: PLOT EVERYTHING TOO TO SEE IF TRULY ACTS LIKE LQR CONTROLLER
    '''



if __name__ == '__main__':
    test()