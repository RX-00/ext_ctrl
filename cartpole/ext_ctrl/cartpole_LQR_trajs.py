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

    def __init__(self, env_id):
        # object member variables
        self.env_id = env_id
        self.env = gym.make(env_id, render_mode="human")

    def run_sim(self):
        self.env.reset()
        for i in range(500):
            print(i)

    def collect_traj(self):
        print("TODO")



if __name__ == "__main__":
    env_id = "NominalCartpole"
    nomCartpoleLQRTrajs = TrajCollector(env_id)
