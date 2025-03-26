import torch
import os
from datetime import datetime
import numpy as np
import torch.nn.functional as F

from learn_utils.models import actor, critic, discriminator
from learn_utils.replay_buffer import replay_buffer
from learn_utils.normalizer import normalizer
from learn_utils.her import her_sampler

"""
GOAL-GAIL with HER

"""
class Goal_gail_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params

        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)

        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)

        # create discriminator network
        self.discriminator = discriminator(env_params)

        # Load the model if required
        if args.load_path != None:
            o_mean, o_std, g_mean, g_std, load_actor_model, load_critic_model = torch.load(self.args.load_path, map_location=lambda storage, loc: storage, weights_only=False)
            self.actor_network.load_state_dict(load_actor_model)
            self.critic_network.load_state_dict(load_critic_model)
            print('Load pretrained model from: {}'.format(self.args.load_path))

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # pass all networks to device
        self.device = torch.device(self.args.device)
        print('Device selected: {}'.format(self.device))
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        self.actor_target_network.to(self.device)
        self.critic_target_network.to(self.device)
        self.discriminator.to(self.device)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr = self.args.lr_disc)

        # Load the expert dataset
        self.demo_path = 'RL_trained_agents/' + self.args.env_name + '/RL_expert_record_suboptimal_' + str(self.args.demo_length) + 'demos.npz'

        # create her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.unwrapped.compute_reward)

        # create the replay buffer
        self.real_buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)

        self.expert_buffer = replay_buffer(self.env_params, self.args.demo_length*self.env_params['max_timesteps'], self.her_module.sample_her_transitions)

        # self.all_buffer = replay_buffer(self.env_params, self.args.demo_length*self.env_params['max_timesteps'] + self.args.buffer_size, self.her_module.sample_her_transitions,
                                        # fifo_offset = self.args.demo_length)

        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

        # inisitalise gail weight and anneal
        self.gail_weight = self.args.gail_weight
        gail_anneal_name = ''
        if self.args.enable_gail_anneal:
            print('Enable gail weight annealing')
            self.anneal_co = 1.0
            gail_anneal_name = 'anneal'

        # initialise demo weight and anneal
        self.batch_demo_weight = self.args.batch_demo_weight
        demo_anneal_name = ''
        if self.args.enable_demo_anneal:
            print('Enable demo weight annealing')
            self.anneal_co = 1.0
            demo_anneal_name = 'anneal'

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # path to save the model
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = os.path.join(self.model_path, 'seed_' + str(self.args.seed))
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.save_name_suffix = 'goalgail_suboptimal_gailweight' + str(self.args.gail_weight) + gail_anneal_name + '_rewardD_demo' + str(self.args.demo_length) + demo_anneal_name

        self.best_success_rate = 0.0


    def learn(self):
        """
        train the network

        """

        best_success_rate = -1
        success_rate = self._eval_agent()
        success_rate_all = [success_rate]
        print('Initial success rate: {}'.format(success_rate))

        # load demonstration episodes
        print('[{}] Loading {} demos from {}'.format(datetime.now(), self.args.demo_length, self.demo_path))
        demo_loaded = np.load(self.demo_path)
        mb_obs = demo_loaded['obs']
        mb_ag = demo_loaded['ag']
        mb_g = demo_loaded['g']
        mb_actions = demo_loaded['action']

        # convert them into arrays
        mb_obs = np.array(mb_obs)
        mb_ag = np.array(mb_ag)
        mb_g = np.array(mb_g)
        mb_actions = np.array(mb_actions)

        # store the episodes
        self.expert_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
        self.real_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions], is_demo = True)
        self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
        # self.expert_buffer.prepare_match_sample_data()

        # start to collect new samples
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation, info = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_norm_tensor = self._preproc_inputs(obs, g)
                            pi = self.actor_network(input_norm_tensor)
                            action = self._action_postpro(pi)
                        # feed the actions into the environment
                        observation_new, reward, _, _, info = self.env.step(action)
                        #self.env.render()
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
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
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)

                # store the episodes
                self.real_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                # self.all_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions], is_demo = False)

                # Update the discriminator
                for dis_batch in range(self.args.n_dis_batches):
                    self._update_discriminator()

                # Update the networks
                for batch in range(self.args.n_batches):
                    # train the network
                    self._update_network_ddpg_gail()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent()
            success_rate_all.append(success_rate)
            np.save(self.model_path + '/eval_success_rates_' + self.save_name_suffix + '.npy', success_rate_all)

            # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
            #             self.model_path + '/model_last.pt')

            if success_rate >= self.best_success_rate:
                self.best_success_rate = success_rate
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/model_best_' + self.save_name_suffix + '.pt')

            print('[{}] epoch: {}, eval success rate: {:.3f}, best success rate: {:.3f}'.format(datetime.now(), epoch, success_rate, self.best_success_rate))

            # anneal gail weight if necessary
            if self.args.enable_gail_anneal and epoch > 0:
                # self.anneal_co = np.power(1.0 - success_rate, epoch)
                self.anneal_co = 1.0 - success_rate
                self.gail_weight = self.args.gail_weight*self.anneal_co

            # anneal demo weight if necessary
            if self.args.enable_demo_anneal and epoch > 0:
                # self.anneal_co = np.power(1.0 - success_rate, epoch)
                self.anneal_co = 1.0 - success_rate
                self.batch_demo_weight = self.args.batch_demo_weight*self.anneal_co


    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs = inputs.to(self.device)
        return inputs

    # this function will choose action for the agent and do the exploration
    def _action_postpro(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])

        # generate random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])

        # choose if to use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_normal_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the discriminator network
    def _update_discriminator(self):
        # sample the episodes
        transitions = self.real_buffer.sample(self.args.dis_batch_size)
        demo_transitions = self.expert_buffer.sample(self.args.dis_batch_size)

        # pre-process the observation and goal
        o, g = self._preproc_og(transitions['obs'], transitions['g'])
        demo_o, demo_g = self._preproc_og(demo_transitions['obs'], demo_transitions['g'])
        o_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)
        inputs_norm= np.concatenate([o_norm, g_norm], axis=1)
        demo_o_norm = self.o_norm.normalize(demo_o)
        demo_g_norm = self.g_norm.normalize(demo_g)
        demo_inputs_norm = np.concatenate([demo_o_norm, demo_g_norm], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        demo_inputs_norm_tensor = torch.tensor(demo_inputs_norm, dtype=torch.float32)
        demo_actions_tensor = torch.tensor(demo_transitions['actions'], dtype=torch.float32)
        inputs_norm_tensor = inputs_norm_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        demo_inputs_norm_tensor = demo_inputs_norm_tensor.to(self.device)
        demo_actions_tensor = demo_actions_tensor.to(self.device)

        # Output of discriminator is [0, 1].
        d_policy = self.discriminator(inputs_norm_tensor, actions_tensor)
        d_demo = self.discriminator(demo_inputs_norm_tensor, demo_actions_tensor)

        # -log(sigmoid(D_pi))) - log(sigmoid(1-D_demo))

        # calculate the loss for the discriminator network
        loss = torch.nn.BCELoss()
        loss_disc = loss(d_policy, torch.zeros((d_policy.shape[0],1)).to(self.device)) + \
                    loss(d_demo, torch.ones((d_demo.shape[0],1)).to(self.device))

        # update the discriminator
        self.disc_optim.zero_grad()
        loss_disc.backward()
        self.disc_optim.step()


    # update the actor and critic network
    def _update_network_ddpg_gail(self):
        # sample the episodes
        transitions = self.real_buffer.sample(self.args.batch_size)
        # num_demo = np.sum(transitions['isdemo'])
        # num_real = self.args.batch_size - num_demo
        # # print('Sampled transitions demo number: {}, real number: {}'.format(num_demo, num_real))
        # self.sample_demo_percentage.append(num_demo/self.args.batch_size)
        # self.sample_real_percentage.append(num_real/self.args.batch_size)

        # demo_size = (int)(self.args.batch_size*self.batch_demo_weight)
        # real_size = self.args.batch_size - demo_size
        # real_transitions = self.real_buffer.sample(real_size)
        # demo_transitions = self.expert_buffer.sample(demo_size)
        # transitions = {k:np.concatenate([real_transitions[k],demo_transitions[k]], axis = 0)
        #                 for k in real_transitions.keys()}

        # pre-process the observation and goal
        o, g = self._preproc_og(transitions['obs'], transitions['g'])
        o_next, g_next = self._preproc_og(transitions['obs_next'], transitions['g'])
        # start to do the update
        obs_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(o_next)
        g_next_norm = self.g_norm.normalize(g_next)
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        inputs_norm_tensor = inputs_norm_tensor.to(self.device)
        inputs_next_norm_tensor = inputs_next_norm_tensor.to(self.device)
        actions_tensor = actions_tensor.to(self.device)
        r_tensor = r_tensor.to(self.device)

        # add GAIL reward,
        # gail_reward_tensor = torch.log(self.discriminator(inputs_norm_tensor, actions_tensor) + 1e-8)
        # gail_reward_tensor = -torch.log(1 - self.discriminator(inputs_norm_tensor, actions_tensor))
        gail_reward_tensor = self.discriminator(inputs_norm_tensor, actions_tensor)
        r_tensor = r_tensor + self.gail_weight*gail_reward_tensor

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma) #50
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()


    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation, info = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, reward, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        return local_success_rate
