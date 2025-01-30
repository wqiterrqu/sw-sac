import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import wandb
import os
from pathlib import Path
import socket
from datetime import datetime
from config import args
from utils import Stack
import metaworld

run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / args.env_name / args.algorithm_name / args.experiment_name
if not run_dir.exists():
    os.makedirs(str(run_dir))
# 启动一个WandB实验
run = wandb.init(config=args,
                 project='meta',
                 notes=socket.gethostname(),
                 name=str(args.env_name) + "_" + str(args.algorithm_name) + "_" +
                      str(args.experiment_name) +
                      "_seed" + str(args.seed),
                 dir=str(run_dir),
                 job_type="training",
                 reinit=True)

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)

obs = env.reset(seed=args.seed)[0]
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                     args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

stack = Stack(size=args.SlideLength, beta=args.beta)
#stack = Stack(size=args.SlideLength)

# Training Loop
total_numsteps = 0
updates = 0
best_success = 0.0
best_reward = 0
reward_factor = args.reward_factor
reward_multiple = args.reward_multiple
tolerator = args.tolerator
best_reward_delta = 1.01
best_reward_interval = args.best_reward_interval
best_reward_interval_count = 0
best_reward_average = 0
MAX_REWARD_MULTIPLE = args.max_reward_multiple
pre_reward = 0
beta = args.beta

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state, _ = env.reset()
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                wandb.log(
                    data={
                        'loss/q_critic_1': critic_1_loss,
                        'loss/q_critic_2': critic_2_loss,
                        'loss/policy_loss': policy_loss,
                        'loss/entropy_loss': ent_loss,
                        'entropy_temprature/alpha': alpha,
                    },
                    step=total_numsteps
                )
                updates += 1
        next_state, reward, done, _, info = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        wandb.log(
            data={
                'reward/step_reward': reward
            },
            step=total_numsteps
        )
        # print("reward:{}      best_reward:{}".format(reward, best_reward))
        # if reward > best_reward and total_numsteps>5000:
        #     best_reward_interval_count += 1
        #     best_reward_average += reward
        #     if best_reward_interval_count >= best_reward_interval:
        #         best_reward_interval_count = 0
        #         best_reward = best_reward_average / best_reward_interval
        #         best_reward_average = 0
        #         reward *= min(reward_multiple, MAX_REWARD_MULTIPLE)
        #         reward_multiple *= reward_factor
        #         print("best_reward: {}".format(best_reward))
        #         wandb.log(
        #             data={
        #                 'reward/best_reward': best_reward,
        #                 'reward/reward_multiple': reward_multiple,
        #                 'reward/reward_factor': reward_factor
        #             },
        #             step=total_numsteps
        #         )
        # elif reward > best_reward * tolerator and total_numsteps>5000:
        #     reward *= min(reward_multiple, MAX_REWARD_MULTIPLE)
        # else:
        #     pass

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        if '_max_episode_steps' in dir(env):
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        elif 'max_path_length' in dir(env):
            mask = 1 if episode_steps == env.max_path_length else float(not done)
        else:
            mask = 1 if episode_steps == 1000 else float(not done)

        if args.RewardType == 'baseline':
            memory.push(state, action, reward, next_state, mask)  # Append transition to memory
        elif args.RewardType == 'sliding_window':
           reward_mem = stack.push(reward)
           memory.push(state, action, reward_mem, next_state, mask)
            # reward_mem = pre_reward * beta + reward * (1 - beta)
            # memory.push(state, action, reward_mem, next_state, mask) # Append transition to memory
            # pre_reward = reward
        else:
            raise NotImplementedError

        state = next_state
        if episode_steps > env._max_episode_steps:
            break
    if total_numsteps > args.num_steps:
        break
    wandb.log(
        data={
            'reward/train_reward': episode_reward
        },
        step=total_numsteps
    )
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))
    # wandb.log({"i_episode": i_episode, "reward": round(episode_reward, 2), "episode_steps": episode_steps, "total_numsteps": total_numsteps})

    if i_episode % args.eval_interval == 0 and args.eval is True:
        eval_reward_list = []
        for _ in range(args.eval_episodes):
            state, _ = env.reset()
            episode_reward = []
            done = False
            for _ in range(env._max_episode_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _, info = env.step(action)
                state = next_state
                episode_reward.append(reward)
            eval_reward_list.append(sum(episode_reward))
        avg_reward = np.average(eval_reward_list)
        # if args.save_checkpoint:
        #     if avg_reward >= best_reward:
        #         best_reward = avg_reward
        # agent.save_checkpoint(checkpoint_path, 'best')
        wandb.log(
            data={
                'reward/test_avg_reward': avg_reward,
            },
            step=total_numsteps
        )

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        # wandb.log({'episodes': episodes, 'average_reward': round(avg_reward, 2)})
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(args.eval_episodes, round(avg_reward, 2)))
        print("----------------------------------------")
run.finish()
env.close()

