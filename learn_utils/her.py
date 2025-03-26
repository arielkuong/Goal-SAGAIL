import numpy as np
import random

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else: # if 'replay_strategy' == 'none', do not reform transitions
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                        for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previous selected Here
        # transitions (defined by her_indexes). Keep original goal for the other
        # transitions.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}

        #val = input('Pause...')
        return transitions

    def sample_normal_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                        for key in episode_batch.keys()}

        # Compute reward since we may have substituted the goal.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}

        return transitions

    # # Function used by sample_her_transitions_prioritized_replay
    # def _sample_proportional(self, rollout_batch_size, batch_size, n_transitions_stored, it_sum, T):
    #     episode_idxs = []
    #     t_samples = []
    #     for _ in range(batch_size):
    #         #self.n_transitions_stored = min(self.n_transitions_stored, self.size_in_transitions)
    #         mass = random.random() * it_sum.sum(0, n_transitions_stored - 1)
    #         idx = it_sum.find_prefixsum_idx(mass)
    #         assert idx < n_transitions_stored
    #         episode_idx = idx//T
    #         assert episode_idx < rollout_batch_size
    #         t_sample = idx%T
    #         episode_idxs.append(episode_idx)
    #         t_samples.append(t_sample)
    #
    #     return (episode_idxs, t_samples)
    #
    # def sample_her_transitions_prioritized_replay(self, episode_batch, batch_size_in_transitions,
    #         current_size_in_rollouts, n_transitions_stored, it_min, it_sum, beta):
    #
    #     T = episode_batch['actions'].shape[1]
    #     rollout_batch_size = episode_batch['actions'].shape[0]
    #     batch_size = batch_size_in_transitions
    #
    #     if rollout_batch_size < current_size_in_rollouts:
    #         episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #         t_samples = np.random.randint(T, size=batch_size)
    #     else:
    #         assert beta >= 0
    #         episode_idxs, t_samples = self._sample_proportional(rollout_batch_size, batch_size, n_transitions_stored, it_sum, T)
    #         episode_idxs = np.array(episode_idxs)
    #         t_samples = np.array(t_samples)
    #
    #     weights = []
    #     p_min = it_min.min() / it_sum.sum()
    #     max_weight = (p_min * n_transitions_stored) ** (-beta)
    #
    #     for episode_idx, t_sample in zip(episode_idxs, t_samples):
    #         p_sample = it_sum[episode_idx*T+t_sample] / it_sum.sum()
    #         weight = (p_sample * n_transitions_stored) ** (-beta)
    #         weights.append(weight / max_weight)
    #
    #     weights = np.array(weights)
    #
    #     transitions = {}
    #     for key in episode_batch.keys():
    #         if not key == "td" and not key == "e":
    #             episode_batch_key = episode_batch[key].copy()
    #             transitions[key] = episode_batch_key[episode_idxs, t_samples].copy()
    #
    #     # Select future time indexes proportional with probability future_p. These
    #     # will be used for HER replay by substituting in future goals.
    #     her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
    #
    #     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     future_t = (t_samples + 1 + future_offset)[her_indexes]
    #
    #     #if replay_strategy == 'final':
    #     #    future_t[:] = T
    #
    #     # Replace goal with achieved goal but only for the previously-selected
    #     # HER transitions (as defined by her_indexes). For the other transitions,
    #     # keep the original goal.
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    #     transitions['g'][her_indexes] = future_ag
    #
    #     # Re-compute reward since we may have substituted the goal.
    #     transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
    #
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #                     for k in transitions.keys()}
    #
    #     assert(transitions['actions'].shape[0] == batch_size_in_transitions)
    #
    #     idxs = episode_idxs * T + t_samples
    #
    #     return (transitions, weights, idxs)
