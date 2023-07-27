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
            self.layer_init(nn.Linear(np.array(state_dim).prod(), hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, 1), std=1.0),
        )
        # policy
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.array(state_dim).prod(), hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hl_size, np.prod(action_dim)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
    
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
        self.advantages = torch.zeros_like(self.rewards).to(device)
        self.returns = self.advantages + self.values
    
    def flatten_batch(self):
        self.b_obs = self.obs.reshape((-1,) + self.state_dim)
        self.b_logprobs = self.logprobs.reshape(-1)
        self.b_actions = self.actions.reshape((-1,) + self.action_dim)
        self.b_advantages = self.advantages.reshape(-1)
        self.b_returns = self.returns.reshape(-1)
        self.b_values = self.values.reshape(-1)

    def update(self, batch_size, update_epochs, minibatch_size, clip_coef, norm_adv,
               clip_vloss, ent_coef, vf_coef, max_grad_norm, target_kl):
        # flatten the batch
        self.flatten_batch()

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(self.b_obs[mb_inds], 
                                                                                   self.b_actions[mb_inds])
                logratio = newlogprob - self.b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = self.b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
                    v_clipped = self.b_values[mb_inds] + torch.clamp(
                        newvalue - self.b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                self.optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = self.b_values.cpu().numpy(), self.b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var

    
    def save(self, path):
        torch.save(self.agent.state_dict(), path)

    def load(self, path):
        self.agent.load_state_dict(torch.load(path, map_location=lambda storage, loc : storage))