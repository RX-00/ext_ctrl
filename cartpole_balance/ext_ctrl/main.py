# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import ext_ctrl_envs
from sync_vec_env2 import SyncVectorEnv2

NUM_TRAJS = 933

'''
========================================
Helper function for the reward function
========================================
'''
def calc_width_curve_weight(weight_w, weight_c, trajs):
    return (weight_w / (trajs.max() - trajs.min()))**weight_c


'''
=======================================
Function to select a random trajectory
=======================================
'''
def sample_rand_traj():
    # traj file numbering trackers
    '''
    NOTE: has to be the same as in the cartpole_LQR_trajs.py trajectory
            collector program
    '''

    j = random.randint(int(NUM_TRAJS * 0/3), int((NUM_TRAJS - 1) * 3/3))

    traj_file_path = '/home/robo/ext_ctrl/cartpole_balance/ext_ctrl/traj/trajs/'
    traj_file_path = (traj_file_path + 'traj_' + str(j) + '.npz') # 82 is a good demo case
    
    npzfile = np.load(traj_file_path)

    return npzfile

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments 
    parser.add_argument("--env-id", type=str, default="NominalCartpole", # og: NominalCartpole
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e8),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=500,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2, # 4, lower minibatch means using more data of a single episode batch
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--hl-size", type=float, default=64, # 64
        help="the hidden layer size for the NNs")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        # NOTE: modified this (Record Episode Stats) to acomodate for step_traj_track
        env = gym.wrappers.RecordEpisodeStatistics(env)
        #if capture_video:
            #if idx == 0:
            #    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hl_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

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


class Agent_(nn.Module):
    def __init__(self, envs, hl_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(4).prod(), hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(4).prod(), hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, hl_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hl_size, np.prod(1)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(1)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        # make sure the dimension of a single action_mean is torch.tensor([[ ]])
        if len(action_mean.shape) < 2:
            action_mean = torch.tensor([[action_mean.item()]])
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)



def train():
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    dir = "cartpole_balance_pretrained"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = dir + '/' + "NominalCartpole" + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + "PPO_{}_{}.pth".format("NominalCartpole", 0)
    print("Checkpoint for pretrained policies path: " + path)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SyncVectorEnv2(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.hl_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # weights for reward function
    weight_h = 1 # determines max reward val
    weight_w = 2 # determines how close tracking needs to be to start rewarding -> higher = closer requirements
    weight_c = 2 # determines how much reward to give when getting close to perfect tracking -> lower = need to be closer tracking to get full reward

    num_highscore = 0
    num_forgets = 0
    num_traj = 0

    for update in range(1, num_updates + 1):
        # getting sample trajectory
        # select the trajectory to determine reward with
        npzfile = sample_rand_traj()
        num_traj += 1

        # recorded trajectories
        xs         = npzfile['xs']
        x_dots     = npzfile['x_dots']
        thetas     = npzfile['thetas']
        theta_dots = npzfile['theta_dots']
        us         = npzfile['us']        
        
        # calculate intermediate weights for reward function
        w_x = calc_width_curve_weight(weight_w, weight_c, xs)
        w_x_dot = calc_width_curve_weight(weight_w, weight_c, x_dots)
        w_theta = calc_width_curve_weight(weight_w, weight_c, thetas)
        w_theta_dot = calc_width_curve_weight(weight_w, weight_c, theta_dots)
        w_u = calc_width_curve_weight(weight_w, weight_c, us)

        '''
        NOTE: Make sure it lines up with how ob vector is put together:
                [x, theta, x_dot, theta_dot] + u
        '''
        interm_weights = [w_x, w_theta, w_x_dot, w_theta_dot, w_u]

        for i in range(args.num_envs):
            sys_qpos = envs.envs[i].unwrapped.data.qpos
            sys_qvel = envs.envs[i].unwrapped.data.qvel
            sys_qpos[0] = xs[0]
            sys_qpos[1] = thetas[0]
            sys_qvel[0] = x_dots[0]
            sys_qvel[1] = theta_dots[0]
            envs.envs[i].set_state(sys_qpos, sys_qvel)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            # TODO:
            #       "need to modify SyncVectorEnv, maybe need to make your own"
            # OR
            #       calculate reward here
            interm_weights[4] = step # hack for time, TODO: figure out why the env don't update function in runtime
            next_obs, reward, terminated, truncated, infos = envs.step_traj_track(action.cpu().numpy(),
                                                                                  xs[step],
                                                                                  x_dots[step],
                                                                                  thetas[step],
                                                                                  theta_dots[step],
                                                                                  us[step],
                                                                                  weight_h,
                                                                                  interm_weights)
            
                                                                                  
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None or len(info) == 0:
                    continue
                #print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                #writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                #writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            
            # NOTE: maybe have the policy learn K?
            # NOTE: maybe have a reward function exp squared around 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if info is None or len(info) == 0:
            continue
        else:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            print("Steps per sec:", int(global_step / (time.time() - start_time)), " trajs: ", int(num_traj))
        # save model weights
        if update % 10 == 0:
            print("Saving agent actor model weights...")
            torch.save(agent.state_dict(), path)
            print(path)
        
        if int(info['episode']['r']) > 2500:
            num_highscore += 1
        if float(info['episode']['r']) == 0.:
            num_forgets += 1

        if num_highscore > args.num_steps: #or num_forgets > args.num_steps*2:
            print("quitting while the going is good to avoid catastrophic forgetting!")
            torch.save(agent.state_dict(), path)
            break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    envs.close()
    writer.close()

def test():
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    dir = "cartpole_balance_pretrained"

    dir = dir + '/' + "NominalCartpole" + '/'

    path = dir + "W_2hl_64n_2nmb_4punishrwrd.pth" # best performing one!
    #path = dir + "PPO_{}_{}.pth".format("NominalCartpole", 0)
    print("Checkpoint for pretrained policies path: " + path)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SyncVectorEnv2(
        [make_env(args.env_id, i, True, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.hl_size).to(device)
    agent.load_state_dict(torch.load(path, map_location=lambda storage, loc : storage))

    for i in range(10):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size

        # select the trajectory to determine reward with
        npzfile = sample_rand_traj()

        # recorded trajectories
        xs         = npzfile['xs']
        x_dots     = npzfile['x_dots']
        thetas     = npzfile['thetas']
        theta_dots = npzfile['theta_dots']
        us         = npzfile['us']        

        for i in range(args.num_envs):
            sys_qpos = envs.envs[i].unwrapped.data.qpos
            sys_qvel = envs.envs[i].unwrapped.data.qvel
            sys_qpos[0] = xs[0]
            sys_qpos[1] = thetas[0]
            sys_qvel[0] = x_dots[0]
            sys_qvel[1] = theta_dots[0]
            envs.envs[i].set_state(sys_qpos, sys_qvel)

        for step in range(0, args.num_steps):
            
            global_step += 1 * args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                                                                                
            next_obs = torch.Tensor(next_obs).to(device)

    envs.close()


def test_nonnominal():
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    dir = "cartpole_balance_pretrained"

    dir = dir + '/' + "NominalCartpole" + '/'

    path = dir + "W_2hl_64n_2nmb_4punishrwrd.pth" # best performing one!
    #path = dir + "PPO_{}_{}.pth".format("NominalCartpole", 0)
    print("Checkpoint for pretrained policies path: " + path)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #envs = SyncVectorEnv2(
    #    [make_env(args.env_id, i, True, run_name, args.gamma) for i in range(args.num_envs)]
    #)
    #assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    env_id = "NonnonimalCartpole"
    env = gym.make(env_id, render_mode="human")

    #agent = Agent(envs, args.hl_size).to(device)
    agent = Agent_(env, args.hl_size).to(device)
    agent.load_state_dict(torch.load(path, map_location=lambda storage, loc : storage))

    for i in range(10):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size

        # select the trajectory to determine reward with
        npzfile = sample_rand_traj()

        # recorded trajectories
        xs         = npzfile['xs']
        x_dots     = npzfile['x_dots']
        thetas     = npzfile['thetas']
        theta_dots = npzfile['theta_dots']
        us         = npzfile['us']        

        for i in range(args.num_envs):
            sys_qpos = env.unwrapped.data.qpos
            sys_qvel = env.unwrapped.data.qvel
            sys_qpos[0] = xs[0]
            sys_qpos[1] = thetas[0]
            sys_qvel[0] = x_dots[0]
            sys_qvel[1] = theta_dots[0]
            env.set_state(sys_qpos, sys_qvel)

        for step in range(0, args.num_steps):
            
            global_step += 1 * args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, infos = env.step(action.cpu().numpy())
                                                                                
            next_obs = torch.Tensor(next_obs).to(device)

    #envs.close()
    env.close()


if __name__ == "__main__":
    #train()
    #test()
    test_nonnominal()
