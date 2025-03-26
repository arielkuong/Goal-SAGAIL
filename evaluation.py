import torch
from models import actor
from arguments import get_args
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = 'RL_trained_agents/' + args.env_name + '/model_best_her_RL_seed123.pt'
    o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  'max_timesteps': env._max_episode_steps,
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(actor_model)
    actor_network.eval()

    # drop_count = 0
    total_success_rate = []
    for i in range(args.demo_length):
        per_success_rate = []
        observation, info = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env_params['max_timesteps']):
            with torch.no_grad():
                inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
            per_success_rate.append(info['is_success'])
        total_success_rate.append(per_success_rate)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    total_success_rate = np.array(total_success_rate)
    global_success_rate = np.mean(total_success_rate[:, -1])
    print('average success rate is: {:.3f}'.format(global_success_rate))
