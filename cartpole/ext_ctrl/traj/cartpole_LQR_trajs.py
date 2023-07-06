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

import os

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

    def __init__(self, env_id, r_mode):
        # object member variables
        self.env = gym.make(env_id, render_mode=r_mode)
        
        # state observation of the system
        self.state = self.env.reset()[0] # np.array [x, x_d, theta, theta_d]
        # render the system
        self.env.render()

        # numpy arrays of state variable evolution over time
        self.xs         = np.array(self.state[0])
        self.x_dots     = np.array(self.state[1])
        self.thetas     = np.array(self.state[2])
        self.theta_dots = np.array(self.state[3])

        # flag for if episode is terminated or truncated
        self.done = False

        # episode length (500 default)
        self.ep_len = 500

        '''
        Create the state space matrices for LQR control
        '''
        # reset env to init_state (i.e. 0)
        mujoco.mj_resetDataKeyframe(self.env.unwrapped.model,
                                    self.env.unwrapped.data, 0)
        # we use mj_forward (forward dynamics function) to find the acceleration given
        # the state and all the forces in the system
        mujoco.mj_forward(self.env.unwrapped.model,
                          self.env.unwrapped.data)
        self.env.unwrapped.data.qacc = 0 # Asset that there's no acceleration
        # The inverse dynamics function takes accel as input and compute the forces
        # required to create the acceleration. Uniquely, MuJoCo's fast inverse dyn.
        # takes into accoun all constraint, including contacts
        mujoco.mj_inverse(self.env.unwrapped.model,
                          self.env.unwrapped.data)
        # NOTE: make sure the required forces are achievable by your actuators before
        #       continuing with the LQR controller process

        # Save the position and control setpoints to linearize around
        qpos0 = self.env.unwrapped.data.qpos.copy()
        qfrc0 = self.env.unwrapped.data.qfrc_inverse.copy()

        # Finding actuator values that can create the desired forces
        # for motor actuators (which we use) we can mulitple the control setpoints by
        # the pseudo-inverse of the actuation moment arm
        # NOTE: more elaborate actuators would require finite-differencing to recover
        #       d qfrc_actuator / d u
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(self.env.unwrapped.data.actuator_moment)
        ctrl0 = ctrl0.flatten() # save the ctrl setpoint

        nu = self.env.unwrapped.model.nu # Alias for # of actuators
        nv = self.env.unwrapped.model.nv # Alias for # of DoFs
        
        # R matrix : system state cost
        self.R = np.eye(nu)
        # Q matrix : control cost (weighted for x and theta)
        self.Q = np.array([[10,  0,   0,  0 ],
                           [ 0,  1,   0,  0 ],
                           [ 0,  0,  10,  0 ],
                           [ 0,  0,   0,  1 ]])
        
        # Before we solve for the LQR controller, we need the A and B matrices. 
        # These are computed by MuJoCo's mjd_transitionFD function which computes 
        # them using efficient finite-difference derivatives, exploiting the 
        # configurable computation pipeline to avoid recomputing quantities which
        # haven't changed.
        self.A = np.zeros((2*nv, 2*nv))
        self.B = np.zeros((2*nv, nu))
        epsilon = 1e-6
        flg_centered = True
        mujoco.mjd_transitionFD(self.env.unwrapped.model, self.env.unwrapped.data,
                                epsilon, flg_centered, self.A, self.B, None, None)
        
        # Solve discrete Riccati equation.
        self.P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)

        # Compute the feedback gain matrix K.
        self.K = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A

    '''
    Apply LQR controller
    '''
    def apply_LQR_ctrlr(self, x):
        u = -np.dot(self.K, x)
        return u
    

    '''
    Run simulation without collecting reference trajectories
    '''
    def run_sim(self, useCtrlr=True):
        self.env.reset()
        u = 0

        # NOTE: init pos of the pendulum cannot be larger magnitude than 0.45
        # mujoco.mj_resetDataKeyframe(self.env.unwrapped.model, self.env.unwrapped.data, 2)

        for i in range(self.ep_len):

            if useCtrlr == True:
                u = self.apply_LQR_ctrlr(self.state)

            self.state, reward, terminated, truncated, info = self.env.step(action=u)

            # record state vector
            self.xs         = np.append(self.xs,         self.state[0])
            self.x_dots     = np.append(self.x_dots,     self.state[1])
            self.thetas     = np.append(self.thetas,     self.state[2])
            self.theta_dots = np.append(self.theta_dots, self.state[3])


    '''
    Run sim and collecting reference trajectories
    '''
    def run_sim_collect_traj(self):
        '''
        Varying the initial state of the system
        '''
        print("Collecting trajectories...")

        # NOTE: one interval bigger (end val) than needed for purpose of including the prev val
        cart_positions = np.arange(-1.8, 1.9, 0.1)  # NOTE: max and min of the railings of the cartpole
        pend_positions = np.arange(-1.5, 1.6, 0.1) # NOTE: ~half circle range of pend (top)
        i = 0
        j = 0
        for cart_pos_offset in cart_positions:
            for pend_pos_offset in pend_positions:
                self.env.reset()
                '''
                vary the initial state of the cartpole

                qpos array
                [cart_pos, pend_pos]

                qvel array
                [cart_vel, pend_vel]
                '''
                sys_qpos = self.env.unwrapped.data.qpos
                sys_qvel = self.env.unwrapped.data.qvel
                sys_qpos[0] = cart_pos_offset
                sys_qpos[1] = pend_pos_offset

                self.env.set_state(sys_qpos, sys_qvel)

                for time_step in range(self.ep_len):
                    u = self.apply_LQR_ctrlr(self.state)

                    self.state, reward, terminated, truncated, info = self.env.step(action=u)

                    # record state vector
                    self.xs         = np.append(self.xs,         self.state[0])
                    self.x_dots     = np.append(self.x_dots,     self.state[1])
                    self.thetas     = np.append(self.thetas,     self.state[2])
                    self.theta_dots = np.append(self.theta_dots, self.state[3])

                # saving state vars
                file_path = "/home/robo/ext_ctrl/cartpole/ext_ctrl/traj/trajs/"
                file_path = (file_path + 'traj' + str(i) + '_' + 
                                                  str(j) + '.npz')
                np.savez(file_path, xs=self.xs,
                                    x_dots=self.x_dots,
                                    thetas=self.thetas,
                                    theta_dots=self.theta_dots)
                j = j + 1
            i = i + 1
        print("Finished trajectory collection!")
        print("Number of cart positions recorded: ", i)
        print("Number of pend positions recorded: ", j)


    '''
    Plot data of the state vector evolution over time
    '''
    def plot_state_vector(self):
        fig, axs = plt.subplots(4, 1, constrained_layout=True)
        fig.suptitle('Cartpole state vector', fontsize=16)

        axs[0].plot(self.xs)
        axs[0].set_title('Cart position')
        axs[0].set_xlabel('Time step')
        axs[0].set_ylabel('m')

        axs[1].plot(self.x_dots)
        axs[1].set_title('Cart velocity')
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('m/s')

        axs[2].plot(self.thetas)
        axs[2].set_title('Pendulum angular position')
        axs[2].set_xlabel('Time step')
        axs[2].set_ylabel('Radians')

        axs[3].plot(self.theta_dots)
        axs[3].set_title('Pendulum angular velocity')
        axs[3].set_xlabel('Time step')
        axs[3].set_ylabel('Radians/sec')

        plt.show()



if __name__ == "__main__":
    env_id = "NominalCartpole"

    '''
    NOTE:
    use "depth_array" for offscreen
    use "human" for onscreen render
    '''

    r_mode = "human"
    nomCartpoleLQRTrajs = TrajCollector(env_id, r_mode)
    #nomCartpoleLQRTrajs.run_sim_collect_traj()
    nomCartpoleLQRTrajs.run_sim(useCtrlr=True)
    #nomCartpoleLQRTrajs.plot_state_vector()
    
    '''
    file_path = '/home/robo/ext_ctrl/cartpole/ext_ctrl/traj/trajs/'
    file_path = file_path + 'traj0_0.npz'
    npzfile = np.load(file_path)

    xs         = npzfile['xs']
    x_dots     = npzfile['x_dots']
    thetas     = npzfile['thetas']
    theta_dots = npzfile['theta_dots']
    '''

    # don't forget to close the environment!
    nomCartpoleLQRTrajs.env.close()


