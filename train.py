import numpy as np
import gymnasium as gym
import gymnasium_robotics
import os, sys
from mpi4py import MPI
import random
import torch

from learn_utils.arguments import get_args
from LfD_ddpg_agent import LfD_ddpg_agent
from Goal_gail_agent import Goal_gail_agent
from Goal_sagail_agent import Goal_sagail_agent
from ddpg_agent import ddpg_agent

gym.register_envs(gymnasium_robotics)

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs, info = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'max_timesteps': env._max_episode_steps
            }
    #params['reward_type'] = env._kwargs.reward_type
    print('Env observation dimension: {}'.format(params['obs']))
    print('Env goal dimension: {}'.format(params['goal']))
    print('Env action dimension: {}'.format(params['action']))
    print('Env max action value: {}'.format(params['action_max']))
    print('Env max timestep value: {}'.format(params['max_timesteps']))
    return params

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)

    # create the ddpg agent to interact with the environment
    if args.lfd_strategy == 'lfd_ddpg_her' or args.lfd_strategy == 'lfd_ddpg_her_bc':
        print('Select {} strategy'. format(args.lfd_strategy))
        trainer = LfD_ddpg_agent(args, env, env_params)
    elif args.lfd_strategy == 'goal_gail':
        print('Select Goal_gail strategy')
        trainer = Goal_gail_agent(args, env, env_params)
    elif args.lfd_strategy == 'goal_sagail':
        print('Select Goal_sagail strategy')
        trainer = Goal_sagail_agent(args, env, env_params)
    elif args.lfd_strategy == 'none':
        print('No Lfd algorithm selected, execute normal ddpg+her RL')
        trainer = ddpg_agent(args, env, env_params)
    else:
        print('Error on selecting lfd strategy')
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
