# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import deque
import sys

sys.path.append(r'../')
import numpy as np
import torch
import gym
import argparse
import os
import math
from SACUU.utils import ReplayBuffer, show_sparsity
from SAC import SAC
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn.functional as F
import random
import copy
from scipy.spatial import KDTree


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy: SAC, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    avg_episode = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), False)[0]
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_episode += 1
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")

    # 打开文件以追加模式写入
    with open('result.txt', 'a') as file:
        # 将print语句中的内容重定向到文件中
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}", file=file)

    return avg_reward, avg_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default='exptest')  # Experiment name
    parser.add_argument("--env", default='Ant-v2')  # Environment
    parser.add_argument("--seed", default=0, type=int)  # Seed
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)  # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--lr", default=1e-3, type=float)  # Learning rate
    parser.add_argument("--actor_update_frequency", default=1, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--critic_target_update_frequency", default=1, type=int)  # Frequency of target networks updates
    parser.add_argument("--log_std_bounds", default=[-5, 2])  # std bounds of policy
    parser.add_argument("--hidden_dim", default=256, type=int)  # Num of hidden neurons in each layer
    parser.add_argument("--static_actor", action='store_true', default=False)  # Fix the topology of actor
    parser.add_argument("--static_critic", action='store_true', default=False)  # Fix the topology of critic
    parser.add_argument("--actor_sparsity", default=0.0, type=float)  # Sparsity of actor
    parser.add_argument("--critic_sparsity", default=0.0, type=float)  # Sparsity of critic
    parser.add_argument("--delta", default=10000, type=int)  # Mask update interval
    parser.add_argument("--zeta", default=0.5, type=float)  # Initial mask update ratio
    parser.add_argument("--random_grow", action='store_true', default=False)  # Use random grow scheme
    parser.add_argument("--nstep", default=1, type=int)  # N-step
    parser.add_argument("--delay_nstep", default=0, type=int)  # Delay of using N-step
    parser.add_argument("--buffer_max_size", default=int(3e6), type=int)  # Upper bound of buffer capacity
    parser.add_argument("--buffer_min_size", default=int(2e5), type=int)  # Lower bound of buffer capacity
    parser.add_argument("--use_dynamic_buffer", action='store_true', default=True)  # Use dynamic buffer
    parser.add_argument("--buffer_threshold", default=0.2, type=float)  # Threshold of policy distance
    parser.add_argument("--buffer_adjustment_interval", default=int(1e4),
                        type=int)  # How often (time steps) we check the buffer

    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--to_spike', type=str, default='regular')
    parser.add_argument('--spike_ts', type=int, default=10)  # 5->10 能够更好的体现二阶神经元的动力学优势
    parser.add_argument('--ntype', type=str, default='LIF')  # LIF,Ax,Pretrain

    args = parser.parse_args()
    args.T_end = (args.max_timesteps - args.start_timesteps)
    the_dir = 'results_SAC'
    root_dir = './' + the_dir + '/' + args.exp_id + '_' + args.env
    argsDict = copy.deepcopy(args.__dict__)
    del argsDict['seed']
    config_json = json.dumps(argsDict, indent=4)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_json = open(root_dir + '/config.json', 'w')
    file_json.write(config_json)
    file_json.close()
    if not os.path.exists("./" + the_dir):
        os.makedirs("./" + the_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("---------------------------------------")
    print(f"Policy: SAC, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=10,
                     mean_range=(-3, 3),
                     std=math.sqrt(0.15),
                     spike_ts=args.spike_ts,
                     device=device,
                     to_spike=args.to_spike,
                     ntype=args.ntype,
                     )
    COMMENT = "td3-spikeAdeepC-" + args.env + '-' + '-spike_ts-' + str(args.spike_ts) + \
              '-' + str(args.to_spike) + "-encoder-dim-" + str(AC_KWARGS['encoder_pop_dim']) + '-' + args.ntype

    exp_dir = root_dir + '/' + str(args.seed)
    tensorboard_dir = exp_dir + '/tensorboard/'
    model_dir = exp_dir + '/model/'

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    # env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    args.state_dim = state_dim
    args.action_dim = action_dim
    args.max_action = max_action

    writer = SummaryWriter(tensorboard_dir)
    policy = SAC(args, writer, ac_kwargs=AC_KWARGS)

    if args.actor_sparsity > 0:
        print("Training a sparse actor network")
        show_sparsity(policy.actor.state_dict())
    if args.critic_sparsity > 0:
        print("Training a sparse critic network")
        show_sparsity(policy.critic1.state_dict())
        show_sparsity(policy.critic2.state_dict())

    replay_buffer = ReplayBuffer(state_dim, action_dim, args.buffer_max_size)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)[0]]
    recent_eval = deque(maxlen=20)
    best_eval = np.mean(evaluations)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    eval_num = 0

    torch.save(policy.actor.state_dict(), model_dir + 'actor0')
    torch.save(policy.critic1.state_dict(), model_dir + 'critic1')
    torch.save(policy.critic2.state_dict(), model_dir + 'critic2')

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
            action_mean = action
        else:
            action, action_mean = policy.select_action(np.array(state))
            action = (
                action
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer

        replay_buffer.add(state, action, next_state, reward, done_bool, action_mean,
                          episode_timesteps >= env._max_episode_steps)
        if args.use_dynamic_buffer and (t + 1) % args.buffer_adjustment_interval == 0:
            if replay_buffer.size == replay_buffer.max_size:
                ind = (replay_buffer.ptr + np.arange(8 * args.batch_size)) % replay_buffer.max_size
            else:
                ind = (replay_buffer.left_ptr + np.arange(8 * args.batch_size)) % replay_buffer.max_size
            batch_state = torch.FloatTensor(replay_buffer.state[ind]).to(device)
            batch_action_mean = torch.FloatTensor(replay_buffer.action_mean[ind] / max_action).to(device)

            states = batch_state.cpu().numpy()
            actions = batch_action_mean.cpu().numpy()
            data = np.hstack([states, actions])
            kd_tree = KDTree(data)
            k = 1
            with torch.no_grad():
                current_action = policy.actor(batch_state).mean / max_action
                ## Get the nearest neighbor
                key = torch.cat([batch_state, current_action], dim=1).detach().cpu().numpy()
                _, idx = kd_tree.query(key, k=[k], workers=-1)
                ## Calculate the regularization
                nearest_neighbour = (torch.tensor(data[idx][:, :, -action_dim:]).squeeze(dim=1).to(device)).type(
                    current_action.dtype)
                distance = F.mse_loss(current_action, nearest_neighbour.type(current_action.dtype)) / 2
                # distance = F.mse_loss(current_action, batch_action_mean) / 2
            writer.add_scalar('buffer_distance', distance, t)
            if distance > args.buffer_threshold and replay_buffer.size > args.buffer_min_size:
                replay_buffer.shrink()
            writer.add_scalar('buffer_size', replay_buffer.size, t)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer=replay_buffer, batch_size=args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_result, avg_episode = eval_policy(policy, args.env, args.seed)
            if t >= args.start_timesteps:
                policy.current_mean_reward = eval_result / avg_episode
            evaluations.append(eval_result)
            recent_eval.append(eval_result)
            writer.add_scalar('reward', eval_result, eval_num)
            eval_num += 1
            if np.mean(recent_eval) > best_eval:
                best_eval = np.mean(recent_eval)
                torch.save(policy.actor.state_dict(), model_dir + 'actor')
                torch.save(policy.critic1.state_dict(), model_dir + 'critic1')
                torch.save(policy.critic2.state_dict(), model_dir + 'critic2')
                if args.actor_sparsity > 0: torch.save(policy.actor_pruner.backward_masks, model_dir + 'actor_masks')
                if args.critic_sparsity > 0: torch.save(policy.critic_pruner.backward_masks, model_dir + 'critic_masks')
    writer.close()


if __name__ == "__main__":
    main()
