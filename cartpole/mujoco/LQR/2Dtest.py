import numpy as np

from collections import deque

import matplotlib.pyplot as plt
#%matplotlib inline # uncomment for Jupyter notebook

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
import gym

from scipy import linalg

'''
LQR Controller
'''
# constants and properties of the system
# NOTE: be sure to make sure these are in line with the .xml mujoco model
g = 9.81
lp = 1.0
mp = 0.1
mc = 1.0

# state matrix
a1 = (-12*mp*g) / (13*mc+mp)
a2 = (12*(mp*g + mc*g)) / (lp*(13*mc + mp))
A = np.array([[0, 1, 0,  0],
              [0, 0, a1, 0],
              [0, 0, 0,  1],
              [0, 0, a2, 0]])

# input matrix
b1 = 13 / (13*mc + mp)
b2 = -12/ (lp*(13*mc + mp))
B = np.array([[0 ],
              [b1],
              [0 ],
              [b2]])

R = np.eye(1, dtype=int) * 10     # choose R (weight for input), we want input to be min.
Q = np.array([[10,  0,  0,  0  ],
              [ 0,  1,  0,  0  ],
              [ 0,  0, 10,  0  ],
              [ 0,  0,  0,  1  ]])     # choose Q (weight for state)

# Solves the continuous-time algebraic Riccati equation (CARE).
P = linalg.solve_continuous_are(A, B, Q, R)

# Calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))

# Function to apply control gain K, returning the direction of u, and u
def apply_ctrlr(K, x):
    # feedback controller
    u = -np.dot(K, x) # u = -Kx, where x is the state vector
    if u > 0:
        return 1, u
    else:
        return 0, u

# get environment
env = gym.make('CartPole-v1')
env.env.seed(1)     # seed for reproducibility
state = env.reset()

x_positions = [state[0]]
theta_positions = [state[2]]
us = []

for i in range(500):
    env.render()

    # get force direction (action) and force value (force)
    action, force = apply_ctrlr(K, state)
    print("u: ", force)
    
    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    abs_force = abs(float(np.clip(force, -100, 100)))
    
    # change magnitute of the applied force in CartPole
    env.env.force_mag = abs_force

    # apply action
    state, reward, done, _ = env.step(action)

    x_positions.append(state[0])
    theta_positions.append(state[2])
    us.append(abs_force)


env.close()

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(x_positions)
axs[0].set_title('Cart position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')

fig.suptitle('Cartpole LQR Controller', fontsize=16)

axs[1].plot(theta_positions)
axs[1].set_xlabel('Time step')
axs[1].set_title('Pendulum angular position')
axs[1].set_ylabel('Radians')


plt.show()