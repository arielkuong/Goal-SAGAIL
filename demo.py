import torch
from learn_utils.models import actor
from learn_utils.arguments import get_args
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import imageio

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
    model_path = 'RL_trained_agents/' + args.env_name + '/model_RL_best_seed123.pt'
    o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    # create the environment
    env = gym.make(args.env_name, render_mode = "rgb_array", width = 960, height = 960)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # get the env param
    observation, info = env.reset()
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

    frames = []
    for i in range(args.demo_length):
        observation, info = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env_params['max_timesteps']):
            with torch.no_grad():
                input_norm_tensor = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
                pi = actor_network(input_norm_tensor)
            action = pi.detach().cpu().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, _, info = env.step(action)
            frames.append(env.render())
            obs = observation_new['observation']
            #print('reward: {}'.format(reward))

        #if env.is_on_palm() == False:
        #    print('Object dropped!!!')

        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
    imageio.mimsave(args.env_name + '.gif',frames)
