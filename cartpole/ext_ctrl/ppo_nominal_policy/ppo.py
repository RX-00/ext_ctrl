'''

PPO implementation that assumes the action space of the system is continuous.

For reference please refer to:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

NOTE: Continuous Action-Space

'''

# PyTorch
# NOTE: you must import torch before mujoco or else there's an invalid pointer error
#       - source: https://github.com/deepmind/mujoco/issues/644
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

# Mujoco and custom environments
import gymnasium as gym
import ext_ctrl_envs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
=====================
Rollout Buffer Class
=====================
'''
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_vals = []
        self.is_terminals = []

    def clear(self): # completely clear out rollout buffer
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_vals[:]
        del self.is_terminals[:]


'''
==============================
Gaussian Actor & Critic Class
==============================
'''
# NOTE: 
class GaussActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_dev_init):
        super(GaussActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), 
                                     action_std_dev_init * action_std_dev_init).to(device)
        # Actor network
        self.actor = nn.Sequential(
                                   nn.Linear(state_dim, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, action_dim),
                                   nn.Tanh()
                                   )

        # Critic network
        self.critic = nn.Sequential(
                                    nn.Linear(state_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 1)
                                   )


    def set_action_std_dev(self, new_action_std_dev):
        self.action_var = torch.full((self.action_dim,),
                                     new_action_std_dev * new_action_std_dev).to(device)


    def forward(self):
        raise NotImplementedError # since we used a sequential model


    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsequeeze(dim=0)
        distr = MultivariateNormal(action_mean, cov_mat)

        action = distr.sample()
        action_logprob = distr.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        distr = MultivariateNormal(action_mean, cov_mat)

        # NOTE: if continuous action space is of dim 1, we gotta reshape action
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprob = distr.log_prob(action)
        distr_entropy = distr.entropy()
        state_val = self.critic(state)

        return action_logprob, state_val, distr_entropy


'''
=======================================
PPO Class with Continuous Action-Space
=======================================
'''
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_dev_init=0.6):
        self.action_std_dev = action_std_dev_init
        self.gamma = gamma

    