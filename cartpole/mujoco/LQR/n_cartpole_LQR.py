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
LQR Controller (hand written method)
'''
# constants and properties of the system
# NOTE: be sure to make sure these are in line with the .xml mujoco model
g = 9.81
lp = 1.0
mp = 0.1
mc = 1.0

# state transition matrix
a1 = (-12*mp*g) / (13*mc+mp)
a2 = (12*(mp*g + mc*g)) / (lp*(13*mc + mp))
A = np.array([[0, 1, 0,  0],
              [0, 0, a1, 0],
              [0, 0, 0,  1],
              [0, 0, a2, 0]])

# control transition matrix
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
#print(K)

def apply_ctrlr(K, x):
    u = -np.dot(K, x)
    return u

# storing ctrl inputs
us = [np.array(0)]



'''
LQR Controller (mjData method)

Here we use the desired position of the system (state vector = 0 vector) as
the setpoint to linearize around to get our state transition matrix A = df/dx,
where f is some nonlinear dynamics.

We use inverse dynamics to find the best control u to linearize around to find
the control transition matrix B = df/du
'''
# set sys model to init_state
mujoco.mj_resetDataKeyframe(env.unwrapped.model, env.unwrapped.data, 1)
# we use mj_forward (forward dynamics function) to find the acceleration given
# the state and all the forces in the system
mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
env.unwrapped.data.qacc = 0 # Asset that there's no acceleration
# The inverse dynamics function takes accel as input and compute the forces
# required to create the acceleration. Uniquely, MuJoCo's fast inverse dyn.
# takes into accoun all constraint, including contacts
mujoco.mj_inverse(env.unwrapped.model, env.unwrapped.data)
# NOTE: make sure the required forces are achievable by your actuators before
#       continuing with the LQR controller process
#print(env.unwrapped.data.qfrc_inverse)

# Save the position and control setpoints to linearize around
qpos0 = env.unwrapped.data.qpos.copy()
qfrc0 = env.unwrapped.data.qfrc_inverse.copy()

# Finding actuator values that can create the desired forces
# for motor actuators (which we use) we can mulitple the control setpoints by
# the pseudo-inverse of the actuation moment arm
# NOTE: more elaborate actuators would require finite-differencing to recover
#       d qfrc_actuator / d u
ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(env.unwrapped.data.actuator_moment)
ctrl0 = ctrl0.flatten() # save the ctrl setpoint


# Choosing R
nu = env.unwrapped.model.nu # Alias for the number of actuators
R = np.eye(nu)

# Choosing Q
nv = env.unwrapped.model.nv # Alias for number of DoFs
# To determine Q we'll be constructing it as a sum of two terms
#   term 1: a balancing cost that will keep the CoM over the cart
#           described by kinematic Jacobians which map b/w joint
#           space and global Cartesian positions (computed analytically)
#   term 2: a cost for joints moving away from their initial config

# Calculating term 1
mujoco.mj_resetData(env.unwrapped.model, env.unwrapped.data)
env.unwrapped.data.qpos = qpos0
mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)

jac_pole = np.zeros((3, nv)) # make Jacobian for pole
mujoco.mj_jacSubtreeCom(env.unwrapped.model, env.unwrapped.data,
                        jac_pole, env.unwrapped.model.body('pole_1').id)

jac_cart = np.zeros((3, nv)) # make Jacobian for cart
mujoco.mj_jacBodyCom(env.unwrapped.model, env.unwrapped.data,
                     jac_cart, None, env.unwrapped.model.body('cart').id)

jac_diff = jac_pole - jac_cart
Q_balance_cost = jac_diff.T @ jac_diff # remember, its a quadratic cost
print(Q_balance_cost)

# Calculating term 2
# indices of relevant sets of joints 
# (here we have pole hinge_1: id 1 & cart slider: id 2)
joint_names = [env.unwrapped.model.joint(i).name
               for i in range(env.unwrapped.model.njnt)]
joint_addrs = [env.unwrapped.model.joint(name).dofadr[0]
                for name in joint_names]
#print('joint names: ', joint_names)
#print('joint addrs: ', joint_addrs)

Q_joints_mv_cost = np.eye(nv)
Q_joints_mv_cost = Q = np.array([[ 0,  0,  0,  0  ],
                                 [ 0,  1,  0,  0  ],
                                 [ 0,  0,  0,  0  ],
                                 [ 0,  0,  0,  1  ]]) 

# Q term
BALANCE_COST = 100
Q = BALANCE_COST * Q_balance_cost + Q_joints_mv_cost

# reset sys model
mujoco.mj_resetData(env.unwrapped.model, env.unwrapped.data)


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