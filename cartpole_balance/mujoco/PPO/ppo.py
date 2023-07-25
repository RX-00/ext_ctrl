'''

PPO implementation that assumes the action space of the system is continuous.

For reference please refer to:
https://spinningup.openai.com/en/latest/algorithms/ppo.html

'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributions.normal import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hl_size=256):
        super().__init__()
        # value function
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(state_dim, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, 1), std=1.0),
        )
        # policy
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(state_dim, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, action_dim, std=0.01)),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma,
                 K_epochs, eps_clip, hl_size, num_steps, num_envs):
        # NOTE: state_dim should come from: np.array(envs.single_observation_space.shape).prod()
        #       action_dim should come from: np.array(envs.single_action_space.shape).prod()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

        self.agent = ActorCritic(state_dim, action_dim, hl_size).to(device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=lr_actor, eps=1e-5)

        # batch storage
        self.obs = torch.zeros((num_steps, num_envs) + state_dim).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + state_dim).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

    
    def update(self):
    

    def flatten_batch(self):
        

    
    def save(self, path):
        torch.save(self.agent.state_dict(), path)

    def load(self, path):
        self.agent.load_state_dict(torch.load(path, map_location=lambda storage, loc : storage))