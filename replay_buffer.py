import threading
import numpy as np
import sys
from gymnasium_robotics.utils import rotations
from sklearn.neighbors import NearestNeighbors

def goal_distance(goal_a, goal_b, env_name):
    assert goal_a.shape == goal_b.shape
    if env_name == 'FetchPush-v1' or env_name == 'FetchPickAndPlace-v1' or env_name == 'FetchSlide-v1':
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    elif env_name == 'HandManipulateEggRotate-v1' or env_name == 'HandManipulateBlockRotateXYZ-v1' or env_name == 'HandManipulatePenRotate-v1':
        assert goal_a.shape[-1] == 7
        d_rot = np.zeros_like(goal_b[..., 0])
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        return d_rot
    else:
        return np.inf

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, size_in_transitions, sample_func, fifo_offset = 0):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size_in_rollouts = size_in_transitions // self.T
        # memory management
        self.current_size_in_rollouts = 0
        self.current_idx_in_buffer = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        self.fifo_offset = fifo_offset
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),
                        'isdemo': np.empty([self.size_in_rollouts, self.T]),
                        }
        # thread lock
        self.lock = threading.Lock()

        self.current_buffer_obs_flat = []
        self.current_buffer_g_flat = []
        self.current_buffer_input_flat = []
        self.match_demo_idxs_episodes = []
        self.match_demo_idxs_timesteps = []

    # store the episode
    def store_episode(self, episode_batch, is_demo = False):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx_fifo(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

            if is_demo:
                self.buffers['isdemo'][idxs] = np.ones(self.T)
            else:
                self.buffers['isdemo'][idxs] = np.zeros(self.T)


    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            assert self.current_size_in_rollouts > 0
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]

        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    # def sample_progressive(self, batch_size, success_rate):
    #     temp_buffers = {}
    #     with self.lock:
    #         assert self.current_size_in_rollouts > 0
    #         for key in self.buffers.keys():
    #             temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]
    #
    #     temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
    #     temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
    #
    #     # Calculate Tmax for progressive sampling
    #     if success_rate < 0.05:
    #         success_rate = 0.05
    #     Tmax = (int)(success_rate * self.T)
    #
    #     # sample transitions
    #     transitions = self.sample_func(temp_buffers, batch_size, Tmax)
    #     return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"

        # Increment consecutively until hit the end
        if self.current_size_in_rollouts+inc <= self.size_in_rollouts:
            idx = np.arange(self.current_size_in_rollouts, self.current_size_in_rollouts+inc)
        elif self.current_size_in_rollouts < self.size_in_rollouts:
            overflow = inc - (self.size_in_rollouts - self.current_size_in_rollouts)
            idx_a = np.arange(self.current_size_in_rollouts, self.size_in_rollouts)
            idx_b = np.random.randint(0, self.current_size_in_rollouts, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size_in_rollouts, inc)

        # Update the replay size
        self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def _get_storage_idx_fifo(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"

        # Increment consecutively until hit the end
        if self.current_idx_in_buffer+inc < self.size_in_rollouts:
            idx = np.arange(self.current_idx_in_buffer, self.current_idx_in_buffer+inc)
            self.current_idx_in_buffer = self.current_idx_in_buffer + inc
        # If hit the end of the buffer, arrange the rest of the buffer, then arrange the overflowed from the beginning
        elif self.current_idx_in_buffer < self.size_in_rollouts:
            overflow = inc - (self.size_in_rollouts - self.current_idx_in_buffer)
            idx_a = np.arange(self.current_idx_in_buffer, self.size_in_rollouts)
            idx_b = np.arange(self.fifo_offset, self.fifo_offset + overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx_in_buffer = self.fifo_offset + overflow

        assert self.current_idx_in_buffer < self.size_in_rollouts, 'Buffer overflowed!'
        # Update the replay size
        self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)

        if inc == 1:
            idx = idx[0]
        return idx

    def clear(self):
        self.current_size_in_rollouts = 0
        self.current_idx_in_buffer = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),
                        }

    def get_length(self):
        return self.current_size_in_rollouts

    def match_episode(self, initial_g, desired_g, env_name, reward_func=None):
        buffer_initial_g = self.buffers['ag'][:self.current_size_in_rollouts,0,:]
        buffer_desired_g = self.buffers['g'][:self.current_size_in_rollouts,0,:]
        initial_g_expand = np.tile(initial_g, [self.current_size_in_rollouts,1])
        desired_g_expand = np.tile(desired_g, [self.current_size_in_rollouts,1])

        initial_g_distance = goal_distance(buffer_initial_g, initial_g_expand, env_name)
        desired_g_distance = goal_distance(buffer_desired_g, desired_g_expand, env_name)
        combined_distance = initial_g_distance + desired_g_distance

        min_distance_idx = int(np.argmin(combined_distance))
        min_distance_transition_ag_next = self.buffers['ag'][min_distance_idx,1:,:]
        min_distance_transition_g = self.buffers['g'][min_distance_idx,:,:]
        min_distance_transition_r = reward_func(min_distance_transition_ag_next, min_distance_transition_g, None)
        ep_r = np.sum(min_distance_transition_r)

        return ep_r, combined_distance[min_distance_idx], min_distance_idx


    # def prepare_match_sample_data(self):
    #     self.current_buffer_obs_flat = self.buffers['obs'][:self.current_size_in_rollouts, :self.T, :].copy()
    #     self.current_buffer_g_flat = self.buffers['g'][:self.current_size_in_rollouts, :self.T, :].copy()
    #     self.current_buffer_obs_flat = np.resize(self.current_buffer_obs_flat, (self.current_size_in_rollouts * self.T, self.env_params['obs']))
    #     self.current_buffer_g_flat = np.resize(self.current_buffer_g_flat, (self.current_size_in_rollouts * self.T, self.env_params['goal']))
    #     self.current_buffer_input_flat = np.concatenate([self.current_buffer_obs_flat, self.current_buffer_g_flat], axis = 1)
    #     # current_buffer_inputs = np.concatenate([current_buffer_obs, current_buffer_g], axis=1)
    #     print('Current buffer inputs shape: {}'.format(self.current_buffer_input_flat.shape))
    #     self.nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.current_buffer_input_flat)


    # def match_sample_absolute(self, observations, goals, reward_func):
    #     # input_obs = observations.copy()
    #     inputs = np.concatenate([observations.copy(), goals.copy()], axis=1)
    #     batch_size = inputs.shape[0]
    #     print('Match sample inputs shape: {}'.format(batch_size))
    #     # assert self.current_buffer_obs_flat.shape[0] > 0, 'Expert buffer obs flatten error'
    #     assert self.current_buffer_input_flat.shape[0] > 0, 'Expert buffer input flatten error'
    #
    #     match_buffer_idxs = []
    #     # match_buffer_distance = []
    #     # match_equal_counts = []
    #     for i in range(batch_size):
    #         min_idx = 0
    #         distance_min = np.inf
    #         distance_equal_count = 0
    #         for j in range(self.current_buffer_input_flat.shape[0]):
    #             # distance = np.linalg.norm(input_obs[i]-current_buffer_obs[j], ord = 2)
    #             distance = (np.square(inputs[i]-self.current_buffer_input_flat[j])).mean()
    #             if distance < distance_min:
    #                 distance_min = distance
    #                 min_idx = j
    #             # elif distance == distance_min:
    #             #     distance_equal_count = distance_equal_count + 1
    #                 # print('Input index:{}, Distance equal demo index:{}'.format(i,j))
    #         match_buffer_idxs.append(min_idx)
    #         # match_buffer_distance.append(distance_min)
    #         # match_equal_counts.append(distance_equal_count)
    #
    #     match_buffer_idxs = np.array(match_buffer_idxs)
    #     # match_buffer_distance = np.array(match_buffer_distance)
    #     # match_equal_counts = np.array(match_equal_counts)
    #     # print('Match buffer idxs: {}'.format(match_buffer_idxs))
    #     # print('Match buffer min distances: {}'.format(match_buffer_distance))
    #     # print('Match equal counts: {}'.format(match_equal_counts))
    #
    #     match_episode_idxs = match_buffer_idxs / self.T
    #     match_episode_idxs = match_episode_idxs.astype(int)
    #     match_timestep_idxs = match_buffer_idxs % self.T
    #     # print('Match episode idxs: {}'.format(match_episode_idxs))
    #     # print('Match timesstep idxs: {}'.format(match_timestep_idxs))
    #
    #     transitions = {}
    #     for key in self.buffers.keys():
    #         transitions[key] = self.buffers[key][match_episode_idxs, match_timestep_idxs]
    #     transitions['obs_next'] = self.buffers['obs'][match_episode_idxs, match_timestep_idxs + 1]
    #     transitions['ag_next'] = self.buffers['ag'][match_episode_idxs, match_timestep_idxs + 1]
    #     transitions['r'] = reward_func(transitions['ag_next'], transitions['g'], None)
    #
    #     # transitions = {key: self.buffers[key][match_episode_idxs, match_timestep_idxs].copy()
    #     #                 for key in self.buffers.keys()}
    #     #
    #     # transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #     #                 for k in transitions.keys()}
    #
    #     # self.match_demo_idxs_episodes.append(match_episode_idxs)
    #     # self.match_demo_idxs_timesteps.append(match_timestep_idxs)
    #     # np.save(model_path + '/record_match_demo_idxs_episodes_batchsize64.npy', self.match_demo_idxs_episodes)
    #     # np.save(model_path + '/record_match_demo_idxs_timesteps_batchsize64.npy', self.match_demo_idxs_timesteps)
    #
    #     return transitions
    #

    # def match_sample_nearest_neighbour(self, observations, goals, reward_func):
    #     # input_obs = observations.copy()
    #     inputs = np.concatenate([observations.copy(), goals.copy()], axis=1)
    #     batch_size = inputs.shape[0]
    #     # print('Match sample inputs shape: {}'.format(batch_size))
    #     # assert self.current_buffer_obs_flat.shape[0] > 0, 'Expert buffer obs flatten error'
    #     assert self.current_buffer_input_flat.shape[0] > 0, 'Expert buffer input flatten error'
    #
    #     # print(self.current_buffer_input_flat.shape)
    #     # print(inputs.shape)
    #     match_buffer_idxs = []
    #     match_buffer_diss = []
    #     match_buffer_diss, match_buffer_idxs = self.nbrs.kneighbors(inputs)
    #
    #     # print(match_buffer_diss)
    #     # print(match_buffer_idxs)
    #
    #     match_buffer_idxs = np.reshape(match_buffer_idxs, [batch_size])
    #     # print(match_buffer_idxs)
    #
    #     match_episode_idxs = match_buffer_idxs / self.T
    #     match_episode_idxs = match_episode_idxs.astype(int)
    #     match_timestep_idxs = match_buffer_idxs % self.T
    #     # print(match_episode_idxs)
    #     # print(match_timestep_idxs)
    #
    #
    #     transitions = {}
    #     for key in self.buffers.keys():
    #         transitions[key] = self.buffers[key][match_episode_idxs, match_timestep_idxs]
    #     transitions['obs_next'] = self.buffers['obs'][match_episode_idxs, match_timestep_idxs + 1]
    #     transitions['ag_next'] = self.buffers['ag'][match_episode_idxs, match_timestep_idxs + 1]
    #     transitions['r'] = reward_func(transitions['ag_next'], transitions['g'], None)
    #
    #     return transitions




# class prioritized_replay_buffer:
#
#     def __init__(self, env_params, size_in_transitions, sample_func, alpha, env_name, fifo_offset = 0):
#         self.env_params = env_params
#         self.T = env_params['max_timesteps']
#         self.size_in_rollouts = size_in_transitions // self.T
#         # memory management
#         self.current_size_in_rollouts = 0
#         self.current_idx_in_buffer = 0
#         self.n_transitions_stored = 0
#         self.sample_func = sample_func
#
#         assert alpha >= 0
#         self._alpha = alpha
#
#         it_capacity = 1
#         self.size_in_transitions = size_in_transitions
#         while it_capacity < size_in_transitions:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#
#         self.env_name = env_name
#         self.fifo_offset = fifo_offset
#
#         # create the buffer to store info
#         self.buffers = {'obs': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['obs']]),
#                         'ag': np.empty([self.size_in_rollouts, self.T + 1, self.env_params['goal']]),
#                         'g': np.empty([self.size_in_rollouts, self.T, self.env_params['goal']]),
#                         'actions': np.empty([self.size_in_rollouts, self.T, self.env_params['action']]),
#                         'td': np.empty([self.size_in_rollouts, self.T]), # accumulated td-error
#                         # 'e': np.empty([self.size_in_rollouts, self.T]), # trajectory energy
#
#                         'isdemo': np.empty([self.size_in_rollouts, self.T]), # accumulated td-error
#
#                         #'ep_num': np.empty([self.size_in_rollouts, self.T, 1]),
#                         #'frame_num': np.empty([self.size_in_rollouts, self.T, 1]),
#                         }
#         # thread lock
#         self.lock = threading.Lock()
#
#     # store the episode
#     def store_episode(self, episode_batch, is_demo = False):
#         mb_obs, mb_ag, mb_g, mb_actions = episode_batch
#         batch_size = mb_obs.shape[0]
#
#         # if dump_buffer:
#         #     if self.env_name in ['HandManipulatePenRotate-v0', \
#         #                            'HandManipulateEggFull-v0', \
#         #                            'HandManipulateBlockFull-v0', \
#         #                            'HandManipulateBlockRotateXYZ-v0']:
#         #         g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
#         #
#         #         # Calculate rotationary energy
#         #         quaternion = mb_ag[:, :, 3:].copy()
#         #         angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
#         #         diff_angle = np.diff(angle, axis=1)
#         #         angular_velocity = diff_angle / delta_t
#         #         rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
#         #         rotational_energy = np.sum(rotational_energy, axis=2)
#         #
#         #         # calculate the potential energy
#         #         height = mb_ag[:, :, 2]
#         #         height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
#         #         height = height[:,1::] - height_0
#         #         potential_energy = g*m*height
#         #
#         #         # calculate the kinetic energy
#         #         pos_diff = np.diff(mb_ag[:, :, :3], axis=1)
#         #         velocity = pos_diff / delta_t
#         #         kinetic_energy = 0.5 * m * np.power(velocity, 2)
#         #         kinetic_energy = np.sum(kinetic_energy, axis=2)
#         #
#         #         # Calculate the trajectory energy for this episode, apply the clip
#         #         energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
#         #         energy_diff = np.diff(energy_totoal, axis=1)
#         #         energy_transition = energy_totoal.copy()
#         #         energy_transition[:,1::] = energy_diff.copy()
#         #         mb_e = energy_transition
#
#         with self.lock:
#             idxs = self._get_storage_idx_fifo(inc=batch_size)
#             # store the informations
#             self.buffers['obs'][idxs] = mb_obs
#             self.buffers['ag'][idxs] = mb_ag
#             self.buffers['g'][idxs] = mb_g
#             self.buffers['actions'][idxs] = mb_actions
#             if is_demo:
#                 self.buffers['isdemo'][idxs] = np.ones(self.T)
#             else:
#                 self.buffers['isdemo'][idxs] = np.zeros(self.T)
#
#             # if dump_buffer:
#             #     self.buffers['e'][idxs] = mb_e
#
#             self.n_transitions_stored += self.T * batch_size
#             self.n_transitions_stored  = min(self.n_transitions_stored, self.size_in_transitions)
#
#             for idx in idxs:
#                 episode_idx = idx
#                 for t in range(episode_idx*self.T, (episode_idx+1)*self.T):
#                     assert (episode_idx+1)*self.T-1 < min(self.n_transitions_stored, self.size_in_transitions)
#                     self._it_sum[t] = self._max_priority ** self._alpha
#                     self._it_min[t] = self._max_priority ** self._alpha
#
#         # print('{} episodes stored in replay buffer'.format(batch_size))
#         #print('it_num size is: {}'.format(self._it_sum.shape[0]))
#         #print('it_min size is: {}'.format(self._it_min.shape[0]))
#         #print('max_priority is: {}'.format(self._max_priority))
#
#     # def dump_buffer(self, epoch):
#     #     for i in range(self.current_size_in_rollouts):
#     #         entry = {"e": self.buffers['e'][i].tolist(), \
#     #                  "td": self.buffers['td'][i].tolist(), \
#     #                  "ag": self.buffers['ag'][i].tolist() }
#     #         with open('buffer_epoch_{0}.txt'.format(epoch), 'a') as file:
#     #              file.write(json.dumps(entry))  # use `json.loads` to do the reverse
#     #              file.write("\n")
#     #
#     #     print("dump buffer")
#
#     # sample the data from the replay buffer
#     def sample(self, batch_size, beta):
#         temp_buffers = {}
#         with self.lock:
#             assert self.current_size_in_rollouts > 0
#             for key in self.buffers.keys():
#                 temp_buffers[key] = self.buffers[key][:self.current_size_in_rollouts]
#
#         temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
#         temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
#
#
#         # sample transitions
#         transitions, weights, idxs = self.sample_func(temp_buffers, batch_size, \
#         self.current_size_in_rollouts, self.n_transitions_stored, self._it_min, self._it_sum, beta)
#
#         for key in (['r', 'obs_next', 'ag_next'] + list(self.buffers.keys())):
#             if not key == 'td' and not key == 'e':
#                 assert key in transitions, "key %s missing from transitions" % key
#
#         return (transitions, weights, idxs)
#
#     # Update priorities of sampled transitions
#     def update_priorities(self, idxes, priorities):
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities.flatten()):
#             assert priority > 0
#             assert 0 <= idx < self.n_transitions_stored
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha
#
#             self._max_priority = max(self._max_priority, priority)
#
#         #print('priorities updated for {} buffer points'.format(len(idxes)))
#         #print('it_num size is: {}'.format(self._it_sum.shape[0]))
#         #print('it_min size is: {}'.format(self._it_min.shape[0]))
#         #print('max_priority is: {}'.format(self._max_priority))
#
#
#     def _get_storage_idx(self, inc=None):
#         inc = inc or 1   # size increment
#         assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"
#
#         # Increment consecutively until hit the end
#         if self.current_size_in_rollouts+inc <= self.size_in_rollouts:
#             idx = np.arange(self.current_size_in_rollouts, self.current_size_in_rollouts+inc)
#         elif self.current_size_in_rollouts < self.size_in_rollouts:
#             overflow = inc - (self.size_in_rollouts - self.current_size_in_rollouts)
#             idx_a = np.arange(self.current_size_in_rollouts, self.size_in_rollouts)
#             idx_b = np.random.randint(0, self.current_size_in_rollouts, overflow)
#             idx = np.concatenate([idx_a, idx_b])
#         else:
#             idx = np.random.randint(0, self.size_in_rollouts, inc)
#
#         # Update the replay size
#         self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)
#
#         if inc == 1:
#             idx = idx[0]
#         return idx
#
#
#     def _get_storage_idx_fifo(self, inc=None):
#         inc = inc or 1   # size increment
#         assert inc <= self.size_in_rollouts, "Batch committed to replay is too large!"
#
#         # Increment consecutively until hit the end
#         if self.current_idx_in_buffer+inc < self.size_in_rollouts:
#             idx = np.arange(self.current_idx_in_buffer, self.current_idx_in_buffer+inc)
#             self.current_idx_in_buffer = self.current_idx_in_buffer + inc
#         # If hit the end of the buffer, arrange the rest of the buffer, then arrange the overflowed from the beginning
#         elif self.current_idx_in_buffer < self.size_in_rollouts:
#             overflow = inc - (self.size_in_rollouts - self.current_idx_in_buffer)
#             idx_a = np.arange(self.current_idx_in_buffer, self.size_in_rollouts)
#             idx_b = np.arange(self.fifo_offset, self.fifo_offset + overflow)
#             idx = np.concatenate([idx_a, idx_b])
#             self.current_idx_in_buffer = self.fifo_offset + overflow
#
#         assert self.current_idx_in_buffer < self.size_in_rollouts, 'Buffer overflowed!'
#         # Update the replay size
#         self.current_size_in_rollouts = min(self.size_in_rollouts, self.current_size_in_rollouts+inc)
#
#         if inc == 1:
#             idx = idx[0]
#         return idx
