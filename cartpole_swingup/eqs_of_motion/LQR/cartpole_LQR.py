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

g = 9.81
lp = 1.0
mp = 0.1
mk = 1.0

# state matrix
a = g/(lp*(4.0/3 - mp/(mp+mk)))
A = np.array([[0, 1, 0, 0],
              [0, 0, a, 0],
              [0, 0, 0, 1],
              [0, 0, a, 0]])

# input matrix
b = -1/(lp*(4.0/3 - mp/(mp+mk)))
B = np.array([[0], [1/mk], [0], [b]])

R = np.eye(1, dtype=int)          # choose R (weight for input)
Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)

# get riccati solver
from scipy import linalg

# solve ricatti equation
P = linalg.solve_continuous_are(A, B, Q, R)

# calculate optimal controller gain
K = np.dot(np.linalg.inv(R),
           np.dot(B.T, P))
print(K)

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

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
    action, force = apply_state_controller(K, state)
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