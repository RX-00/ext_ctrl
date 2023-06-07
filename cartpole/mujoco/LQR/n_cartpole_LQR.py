'''

Program of an LQR controller applied to a mujoco simulation of an
nominal inverted pendulum on a cart system.

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
Mujoco Gymnasium Environment Setup
'''
# Create the env
env_id = "NominalCartpole"
env = gym.make(env_id, render_mode="human")

# State of system and whether or not to render env
# reset() returns a tuple of the observation and nothing
state = env.reset()[0] # np.array([ x, x_dot, theta, theta_dot ])
env.render()

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# recording info of the system simulation
x_positions = [state[0]]
theta_positions = [state[2]]

# for if the system is in a termination or truncation state
done = False


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
print(K)

def apply_ctrlr(K, x):
    u = -np.dot(K, x)
    return u

# storing ctrl inputs
us = [np.array(0)]

print(env.unwrapped.data.qpos)

exit() # just to stop more computation

'''
Simulation

NOTE: step != timestep, please refer to the .xml file for the simulation timestep
      as that would effect the energy in the system.
'''
#while not done: # while loop for training
for i in range(500): # for testing, 500 steps

    u = apply_ctrlr(K, state)

    print("u: ", u)

    # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
    # These represent the next observation, the reward from the step,
    # if the episode is terminated, if the episode is truncated and
    # additional info from the step
    state, reward, terminated, truncated, info = env.step(action=u)

    # record data about system
    x_positions.append(state[0])
    theta_positions.append(state[2])
    us.append(u)

    # End the episode when either truncated or terminated is true
    #  - truncated: The episode duration reaches max number of timesteps
    #  - terminated: Any of the state space values is no longer finite.
    done = terminated or truncated

env.close()


'''
Plotting data
'''
fig, axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(x_positions)
axs[0].set_title('Cart position')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Position')

fig.suptitle('Cartpole x and theta pos', fontsize=16)

axs[1].plot(theta_positions)
axs[1].set_title('Pendulum angular position')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Radians')

axs[2].plot(us)
axs[2].set_title('Force applied on cart')
axs[2].set_xlabel('Time step')
axs[2].set_ylabel('Newtons')

plt.show()