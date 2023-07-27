'''

Continuous action-space PPO implementation based off of cleanrl with learning the
log of the action standard deviation. This is meant to play the cartpole game in
mujoco and feature training and testing features of saved policy weights.

'''

import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from distutils.util import strtobool
from datetime import datetime

import ext_ctrl_envs

from ppo import PPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument("--env-id", type=str, default="NominalCartpole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


'''
=================================
Create environments (and videos)
=================================
'''
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


'''
=======================
Make the policy learn!
=======================
'''
def train(freq_save_model, path):
    print("\n\n\nBeginning training...")
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

    # seeding RNG
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setup env
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space supported"

    ppoAgent = PPO(state_dim=envs.single_observation_space.shape, 
                   action_dim=envs.single_action_space.shape, 
                   lr_actor=args.learning_rate, 
                   lr_critic=args.learning_rate, 
                   gamma=args.gamma,
                   K_epochs=args.update_epochs, 
                   eps_clip=args.clip_coef, 
                   hl_size=256, # hidden layer size 
                   num_steps=args.num_steps, 
                   num_envs=args.num_envs)
    
    # utility variables
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # anneal learning rate if flag true
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            ppoAgent.optimizer.param_groups[0]["lr"] = lr_now

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            # update batch data
            ppoAgent.obs[step] = next_obs
            ppoAgent.dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value = ppoAgent.agent.get_action_and_value(next_obs)
                ppoAgent.values[step] = value.flatten()
            ppoAgent.actions[step] = action
            ppoAgent.logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            ppoAgent.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                #print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap the value if not done (terminated)
        with torch.no_grad():
            next_value = ppoAgent.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(ppoAgent.rewards).to(device)
            # Generalized Advantage Estimation (GAE) is an exponentially-weighted estimator of an
            # advantage function similar to TD(lambda). It substantially reduces the variance of 
            # policy gradient estimates at the expense of bias.
            last_gae_lam = 0    # this is the prev. generalized advantage estimate's lambda value
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps -1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - ppoAgent.dones[t+1]
                    next_values = ppoAgent.values[t+1]
                delta = ppoAgent.rewards[t] + args.gamma * next_values * next_non_terminal - ppoAgent.values[t]
                ppoAgent.advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
            ppoAgent.returns = ppoAgent.advantages + ppoAgent.values

        # optimize the policy (actor) and value (critic) networks
        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = ppoAgent.update(batch_size=args.batch_size, 
                                                                                                            update_epochs=args.update_epochs,
                                                                                                            minibatch_size=args.minibatch_size,
                                                                                                            clip_coef=args.clip_coef,
                                                                                                            norm_adv=args.norm_adv,
                                                                                                            clip_vloss=args.clip_vloss, 
                                                                                                            ent_coef=args.ent_coef,
                                                                                                            vf_coef=args.vf_coef,
                                                                                                            max_grad_norm=args.max_grad_norm,
                                                                                                            target_kl=args.target_kl)
        
        # save model
        if update % freq_save_model == 0:
            print("Saving model at: ", path)
            ppoAgent.save(path)
            print("... model saved")
        
        # record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", ppoAgent.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("Steps per second:", int(global_step / (time.time() - start_time)))
        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    envs.close()
    writer.close()


if __name__ == "__main__":
    freq_save_model = int(10)
    dir = "ppo_pretrained"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = dir + '/' + "NominalCartpole" + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + "PPO_{}_{}.pth".format("NominalCartpole", 0)
    print("Checkpoint for pretrained policies path: " + path)

    train(freq_save_model, path)
