import torch
from models import actor
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
    model_path = 'RL_trained_agents/' + args.env_name + '/model_RL_suboptimal_seed' + str(args.seed) + '.pt'
    o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    # create the environment
    env = gym.make(args.env_name)
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

    print("..........................Start recording..................................")
    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
    succes_demo_count = 0
    demo_total_count = 0

    while succes_demo_count < args.demo_length:
		# reset the rollouts
        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
        # reset the environment
        observation, info = env.reset()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        ep_r = 0
        # start to collect samples
        for t in range(env_params['max_timesteps']):
            with torch.no_grad():
                input_norm_tensor = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
                pi = actor_network(input_norm_tensor)
            action = pi.detach().cpu().numpy().squeeze()
            # feed the actions into the environment
            observation_new, r, _, _, info = env.step(action)
            #self.env.render()
            obs_new = observation_new['observation']
            ag_new = observation_new['achieved_goal']
            ep_r += r # range from -100 to 0
            # append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            # re-assign the observation
            obs = obs_new
            ag = ag_new

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        demo_total_count += 1
        if info['is_success'] == 1 and ep_r != 0:
        # if ep_r != 0:
        	succes_demo_count += 1
        	mb_obs.append(ep_obs)
        	mb_ag.append(ep_ag)
        	mb_g.append(ep_g)
        	mb_actions.append(ep_actions)
        print('Episode {} finished, Success demo counter {}, success = {}, episode reward {}'.format(demo_total_count, succes_demo_count, info['is_success'], ep_r))
    print('Average demo success rate: {}'.format(succes_demo_count/demo_total_count))
    # convert them into arrays
    mb_obs = np.array(mb_obs)
    mb_ag = np.array(mb_ag)
    mb_g = np.array(mb_g)
    mb_actions = np.array(mb_actions)

    numpy_dict = {
        "obs": mb_obs,
        "ag": mb_ag,
        "g": mb_g,
        "action": mb_actions,
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)

    save_path = 'RL_trained_agents/' + args.env_name + '/RL_expert_record_suboptimal_' + str(args.demo_length) + 'demos.npz'
    np.savez_compressed(save_path, **numpy_dict)
