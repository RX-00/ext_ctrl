'''

Program of an LQR controller applied to a mujoco simulation of an
nominal inverted pendulum on a cart system.

This program also features tha ability to record and store various trajectories
based on different initial conditions. This data is later used for learning a
policy based on the behavior of a state-space controller.

'''

# PyTorch
# NOTE: you must import torch before mujoco or else there's an invalid pointer error
#       - source: https://github.com/deepmind/mujoco/issues/644
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import mujoco
import mediapy as media
import matplotlib.pyplot as plt

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs

# getting riccati solver
from scipy import linalg




'''
=======================================
Collecting & Saving Trajectories Class
=======================================
'''
class TrajCollector():

    def __init__(self, env_id, render_mode):
        # object member variables
        self.env = gym.make(env_id, render_mode)
        
        # state observation of the system
        self.state = self.env.reset()[0]
        self.env.render()

        # numpy arrays of state variable evolution over time
        self.x_positions = np.array(self.state[0])
        self.x_positions.append(self.state[1])
        print(self.x_positions)

    
    # run simulation without collecting reference trajectories
    def run_sim(self):
        self.env.reset()
        for i in range(500):
            print(i)


    # run sim whilst collecting reference trajectories
    def run_sim_collect_traj(self):
        print("TODO")



if __name__ == "__main__":
    env_id = "NominalCartpole"
    render_mode = "rgb_array"
    nomCartpoleLQRTrajs = TrajCollector(env_id, render_mode)
