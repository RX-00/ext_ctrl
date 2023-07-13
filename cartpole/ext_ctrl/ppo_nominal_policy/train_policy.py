'''

Training program for the policy to use reinforcement learning to learn the behavior
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
import time
from datetime import datetime

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs

from ppo import PPO


'''
=======================
Make the policy learn!
=======================
'''
def train():
    print("\n\n\nBeginning training...")
    '''
    ----------------------------
    Environment Hyperparameters
    ----------------------------
    '''
    env_id = "NominalCartpole"
    render_mode = "depth_array"           # NOTE: depth_array for no render, human for yes render
    render_mode_num = 2                   # NOTE: 2 for depth_array, 0 for human render
    ep_len_max = 500                      # max timesteps in one episode
    train_timesteps_max = int(3e8)        # training truncated if timesteps > train_timesteps_max

    freq_save_model = int(1e5)            # frequency to save model, units: [num timesteps]
    freq_print_avg_rwrd = ep_len_max * 10 # frequency to print avg reward return, units: [num timesteps]
    freq_log_avg_rwrd = ep_len_max * 2    # frequency to log avg reward return, units: [num timesteps]

    action_std_dev = 0.6                  # initial std dev for action distr (Multivariate Normal, i.e. Gaussian)
    action_std_dev_decay_rate = 0.05      # linearly decay action_std_dev
    min_action_std_dev = 0.1              # can't decay std dev more than this val
    
                                          # frequency to decay action std dev, units: [num timesteps]
    freq_decay_action_std_dev_decay = int (2.5e5)

    print("Gymnasium env: " + env_id)

    env = gym.make(env_id, render_mode_num)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    '''
    --------------------
    PPO Hyperparameters
    --------------------
    '''
    update_timestep = ep_len_max * 4    # update policy every n timesteps
    K_epochs = 80                       # update policy for K epochs in a single PPO update
    eps_clip = 0.2                      # clip param for PPO-Clip Objective Function
    gamma = 0.99                        # discount factor
    lr_actor = 0.0003                   # learning rate for actor NN
    lr_critic = 0.001                   # learning rate for critic NN

    '''
    -----------------------------
    Logging progress on training
    -----------------------------
    '''
    log_dir = "ext_ctrl_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = log_dir + '/' + env_id + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get number of log files in a directory
    curr_num_files = next(os.walk(log_dir))[2]
    run_num = len(curr_num_files)

    # Create new log file for each run
    log_file_name = log_dir + '/ext_ctrl_' + env_id + '_log_' + str(run_num) + '.csv'
    print("Current logging run number for: " + env_id + ' : ', run_num)
    print("----logged file at: " + log_file_name)

    '''
    ------------------------------
    Checkpointing policy learning
    ------------------------------
    '''
    num_pretrained = 0 # NOTE: This determines file name for the weights
    dir = "ext_ctrl_pretrained"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = dir + '/' + env_id + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    checkpoint_path = dir + "ext_ctrl_{}_{}.pth".format(env_id, num_pretrained)
    print("Checkpoint for pretrained policies path: " + checkpoint_path)

    '''
    -------------------
    Training Procedure
    -------------------
    '''
    ppoAgent = PPO(state_dim, action_dim, lr_actor, lr_critic,
                    gamma, K_epochs, eps_clip, action_std_dev)
    
    # for tracking total training time
    start_time = datetime.now().replace(microsecond=0)
    print("\n\nTraining started at : ", start_time)

    # logging file
    log_file = open(log_file_name, "w+")
    log_file.write('episode,timestep,reward\n')

    # variables for tracking
    print_running_rwrd = 0
    print_running_eps = 0
    log_running_rwrd = 0
    log_running_eps = 0
    time_step = 0
    iter_episode = 0

    # main training loop
    while time_step <= train_timesteps_max:
        state = env.reset()[0]
        curr_ep_rwrd = 0

        # select the trajectory to determine reward with
        cart_positions = np.arange(-1.8, 1.9, 0.1)
        pend_positions = np.arange(-1.5, 1.6, 0.1)
        traj_file_path = '/home/robo/ext_ctrl/cartpole/ext_ctrl/traj/trajs/'
        # traj file numbering trackers
        i = 0
        j = 0
       
        for cart_pos_offset in cart_positions:
            for pend_pos_offset in pend_positions:
                traj_file_path = '/home/robo/ext_ctrl/cartpole/ext_ctrl/traj/trajs/'
                traj_file_path = (traj_file_path + 'traj' + str(i) + '_' +
                                                            str(j) + '.npz')
                
                # print("trajectory file path: ", traj_file_path)
                
                '''
                initiate starting state of the next episode to be
                the same as the trajectory we're critiquing with

                NOTE: currently there are no variations on the init velocities
                      (angular and regular)
                '''
                npzfile = np.load(traj_file_path)
                # recorded trajectories
                xs         = npzfile['xs']
                x_dots     = npzfile['x_dots']
                thetas     = npzfile['thetas']
                theta_dots = npzfile['theta_dots']
                # get and set init state for episode
                sys_qpos = env.unwrapped.data.qpos
                sys_qvel = env.unwrapped.data.qvel
                sys_qpos[0] = xs[0]
                sys_qpos[1] = thetas[0]
                env.set_state(sys_qpos, sys_qvel)
                
                for ts in range(1, ep_len_max + 1):
                    # select action from policy
                    action = ppoAgent.sel_action(state)

                    # apply action and trajectory to determine reward
                    # state, reward, done, _, _ = env.step_traj_track(action,
                    #                                                 xs[ts-1],
                    #                                                 x_dots[ts-1],
                    #                                                 thetas[ts-1],
                    #                                                 theta_dots[ts-1])
                    

                    state, reward, done, _, _ = env.step(action)

                    # save rewards and is_terminals
                    ppoAgent.buffer.rewards.append(reward)
                    ppoAgent.buffer.is_terminals.append(done)

                    # tick next time step and record reward to accumulated ep reward
                    time_step += 1
                    curr_ep_rwrd += reward

                    # update PPO agent
                    if time_step % update_timestep == 0:
                        ppoAgent.update()
                        # decay action std dev of output action distribution
                        ppoAgent.decay_action_std_dev(action_std_dev_decay_rate, min_action_std_dev)

                    # write log to logging file
                    if time_step % freq_log_avg_rwrd == 0:
                        # log avg reward til last episode
                        log_avg_rwrd = log_running_rwrd / log_running_eps
                        log_avg_rwrd = round(log_avg_rwrd, 4)

                        log_file.write('{},{},{}\n'.format(iter_episode, time_step, log_avg_rwrd))
                        log_file.flush()

                        # reset log
                        log_running_rwrd = 0
                        log_running_eps = 0

                    # print avg reward
                    if time_step % freq_print_avg_rwrd == 0:
                        # print average reward till last episode
                        print_avg_rwrd = print_running_rwrd / print_running_eps
                        print_avg_rwrd = round(print_avg_rwrd, 2)

                        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(iter_episode, time_step, print_avg_rwrd))

                        print_running_rwrd = 0
                        print_running_eps = 0

                    # save model weights
                    if time_step % freq_save_model == 0:
                        print("Saving model at: ", checkpoint_path)
                        ppoAgent.save(checkpoint_path)
                        print("... model saved")
                        print("Elapsed time: ", datetime.now().replace(microsecond=0) - start_time)

                    # break if episode is terminated or truncated
                    if done:
                        break
                
                # cumulate the episode just finished
                print_running_rwrd += curr_ep_rwrd
                print_running_eps += 1
                log_running_rwrd += curr_ep_rwrd
                log_running_eps += 1
                iter_episode += 1
            
                j += 1
            i += 1

    # Done training loop
    log_file.close()
    env.close()

    # Print how long it all took
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)    



if __name__ == '__main__':
    train()