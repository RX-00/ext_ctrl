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
from torch.distributions import MultivariateNormal
from torch.distributions import Normal

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
    def __init__(self, state_dim, action_dim, action_std_dev_init, isDecay=True):
        super(GaussActorCritic, self).__init__()
        ls = 64
        self.isDecay = isDecay
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_std = 0
        self.action_var = torch.full((action_dim,), 
                                     action_std_dev_init * action_std_dev_init).to(device)
        # Actor network
        self.actor = nn.Sequential(
                                   nn.Linear(state_dim, ls),
                                   nn.Tanh(),
                                   nn.Linear(ls, ls),
                                   nn.Tanh(),
                                   nn.Linear(ls, action_dim),
                                   nn.Tanh()
                                   )

        # Critic network
        self.critic = nn.Sequential(
                                    nn.Linear(state_dim, ls),
                                    nn.Tanh(),
                                    nn.Linear(ls, ls),
                                    nn.Tanh(),
                                    nn.Linear(ls, 1)
                                   )
        
        if action_dim > 1:
            self.action_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
        else:
            self.action_logstd = nn.Parameter(torch.zeros(np.prod(action_dim)))


    def set_action_std_dev(self, new_action_std_dev):
        self.action_var = torch.full((self.action_dim,),
                                     new_action_std_dev * new_action_std_dev).to(device)


    def forward(self):
        raise NotImplementedError # since we used a sequential model


    def act(self, state):
        action_mean = self.actor(state)
        
        if (self.isDecay):
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            distr = MultivariateNormal(action_mean, cov_mat)

        # NOTE: for learning continuous action space standard deviation
        else: # learn action_logstd
            action_logstd = self.action_logstd.expand_as(action_mean)
            self.action_std = torch.exp(action_logstd) 
            distr = Normal(action_mean, self.action_std)

        action = distr.sample()
        action_logprob = distr.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        if (self.isDecay):
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            distr = MultivariateNormal(action_mean, cov_mat)

        # NOTE: for learning continuous action space standard deviation
        else: # learn action_logstd
            action_logstd = self.action_logstd.expand_as(action_mean)
            self.action_std = torch.exp(action_logstd) 
            distr = Normal(action_mean, self.action_std)

        # NOTE: if continuous action space is of dim 1, we gotta reshape action
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprob = distr.log_prob(action)
        distr_entropy = distr.entropy()
        state_val = self.critic(state)

        return action_logprob, state_val, distr_entropy
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


'''
=======================================
PPO Class with Continuous Action-Space
=======================================
'''
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_dev_init=0.6, isDecay=True):
        self.action_std_dev = action_std_dev_init
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        self.buffer = RolloutBuffer()

        self.policy = GaussActorCritic(state_dim, action_dim, action_std_dev_init, isDecay).to(device)
        self.policy_prev = GaussActorCritic(state_dim, action_dim, action_std_dev_init, isDecay).to(device)
        self.policy_prev.load_state_dict(self.policy.state_dict())

        # NOTE: for some reason, you need to make sure all the NN's have the same learning rate,
        #       otherwise the learned parameter will continue to explode
        if (isDecay):
            self.optimizer = torch.optim.Adam([{'params' : self.policy.actor.parameters(), 'lr' : lr_actor},
                                           {'params' : self.policy.critic.parameters(), 'lr' : lr_critic},
                                           {'params' : self.policy.action_logstd, 'lr' : lr_actor}
                                          ])
        else:
            self.optimizer = torch.optim.Adam([{'params' : self.policy.actor.parameters(), 'lr' : lr_actor},
                                           {'params' : self.policy.critic.parameters(), 'lr' : lr_actor},
                                           {'params' : self.policy.action_logstd, 'lr' : lr_actor}
                                          ])
        
        self.MseLoss = nn.MSELoss()

    
    def set_action_std_dev(self, new_action_std_dev):
        self.action_std_dev = new_action_std_dev
        self.policy.set_action_std_dev(new_action_std_dev)
        self.policy_prev.set_action_std_dev(new_action_std_dev)


    def decay_action_std_dev(self, action_std_dev_decay_rate, min_action_std_dev):
        self.action_std_dev = self.action_std_dev - action_std_dev_decay_rate
        self.action_std_dev = round(self.action_std_dev, 4)

        if (self.action_std_dev <= min_action_std_dev):
            self.action_std_dev = min_action_std_dev
            print("----set actor output action std dev to min action std dev : ", self.action_std_dev)
        else:
            print("----set actor output action std dev to : ", self.action_std_dev)
        
        self.set_action_std_dev(self.action_std_dev)

    
    def sel_action(self, state):
        with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_prev.act(state)

                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_vals.append(state_val)

                return action.detach().cpu().numpy().flatten()
        
    
    def update(self):
        # the reward returns are a Monte Carlo estimate
        rewards = []
        discounted_reward = 0

        # Go through the replay buffer
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert lists into tensors
        prev_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        prev_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        prev_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        prev_state_vals = torch.squeeze(torch.stack(self.buffer.state_vals, dim=0)).detach().to(device)

        # Calculate advantages
        advantages = rewards.detach() - prev_state_vals.detach()

        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):

            # Evaluate prev actions and values
            logprobs, state_vals, distr_entropy = self.policy.evaluate(prev_states, prev_actions)

            # Make sure the state values tensor is dimension compatible with rewards tensor
            state_vals = torch.squeeze(state_vals)

            # Finding the policy ratio
            # i.e.
            #   pi_theta
            # -------------
            # pi_theta_prev
            ratios = torch.exp(logprobs - prev_logprobs.detach())

            # Find the final loss of PPO-Clip Objective Function
            # with the use of surrogate loss function
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_vals, rewards) - 0.01 * distr_entropy

            # Take the gradient step (one optimizer since you put both NN's into same class object)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy the new weights into the prev policy for next eval
        self.policy_prev.load_state_dict(self.policy.state_dict())

        # Clear replay buffer
        self.buffer.clear()

    
    def save(self, checkpoint_path):
        torch.save(self.policy_prev.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_prev.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc : storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc : storage))