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
====================
Standard Simulation
====================
NOTE: step != timestep, please refer to the .xml file for the simulation timestep
      as that would effect the energy in the system.
'''
def std_simulation():
    '''
    -----------------------------------
    Mujoco Gymnasium Environment Setup
    -----------------------------------
    '''
    # Create the env
    env_id = "NominalCartpole"
    env = gym.make(env_id, render_mode="human")

    # State of system and whether or not to render env
    # reset() returns a tuple of the observation and nothing
    state = env.reset()[0] # np.array([ x, x_dot, theta, theta_dot ])


    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.shape[0]

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # recording info of the system simulation
    x_positions = [state[0]]
    theta_positions = [state[2]]
    # storing ctrl inputs
    us = [np.array(0)]

    # for if the system is in a termination or truncation state
    done = False


    '''
    -------------------------------
    LQR Controller (mjData method)
    -------------------------------

    Here we use the desired position of the system (state vector = 0 vector) as
    the setpoint to linearize around to get our state transition matrix A = df/dx,
    where f is some nonlinear dynamics.

    We use inverse dynamics to find the best control u to linearize around to find
    the control transition matrix B = df/du
    '''
    # set sys model to init_state
    mujoco.mj_resetDataKeyframe(env.unwrapped.model, env.unwrapped.data, 0)
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
    # NOTE: Why do they use the number of actuators as the dimensions for R
    #       and number of DoFs as the dimensions for Q????
    R = np.eye(nu)

    # Choosing Q
    nv = env.unwrapped.model.nv # Alias for number of DoFs
    # NOTE: this wasn't used
    # To determine Q we'll be constructing it as a sum of two terms
    #   term 1: a balancing cost that will keep the CoM over the cart
    #           described by kinematic Jacobians which map b/w joint
    #           space and global Cartesian positions (computed analytically)
    #   term 2: a cost for joints moving away from their initial config

    # Classic Q matrix
    # weighted for cart pos & pendulum angle
    Q = np.array([[10,  0,   0,  0 ],
                [ 0,  1,   0,  0 ],
                [ 0,  0,  10,  0 ],
                [ 0,  0,   0,  1 ]])

    # Computing gain matrix K
    # Set the initial state and control.
    mujoco.mj_resetData(env.unwrapped.model, env.unwrapped.data)
    env.unwrapped.data.ctrl = ctrl0 # should be 0
    env.unwrapped.data.qpos = qpos0 # should be 0

    #
    # Before we solve for the LQR controller, we need the A and B matrices. 
    # These are computed by MuJoCo's mjd_transitionFD function which computes 
    # them using efficient finite-difference derivatives, exploiting the 
    # configurable computation pipeline to avoid recomputing quantities which
    # haven't changed.
    # 
    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(env.unwrapped.model, env.unwrapped.data,
                            epsilon, flg_centered, A, B, None, None)

    # Solve discrete Riccati equation.
    P = linalg.solve_discrete_are(A, B, Q, R)

    # Compute the feedback gain matrix K.
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    def apply_LQR_ctrlr(K, x):
        u = -np.dot(K, x)
        return u

    # reset sys model
    env.reset()
    # render model
    env.render()

    #while not done: # while loop for training
    for i in range(500): # for testing, 500 steps

        u = apply_LQR_ctrlr(K, state)

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
=================================
Collecting & Saving Trajectories
=================================
'''
def traj_collect():
    print("Collecting trajectory data...")



if __name__ == "__main__":
    traj_collect()
