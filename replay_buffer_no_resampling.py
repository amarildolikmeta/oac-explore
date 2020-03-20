from collections import OrderedDict
from utils.env_utils import get_dim
from gym.spaces import Discrete

import numpy as np


class ReplayBufferNoResampling(object):

    def __init__(
        self,
        max_replay_buffer_size,
        ob_space,
        action_space,

    ):
        """
        The class state which should not mutate
        """

        self._ob_space = ob_space
        self._action_space = action_space

        ob_dim = get_dim(self._ob_space)
        ac_dim = get_dim(self._action_space)

        self._max_replay_buffer_size = max_replay_buffer_size

        """
        The class mutable state
        """

        self._observations = np.zeros((max_replay_buffer_size, ob_dim))

        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, ob_dim))
        self._actions = np.zeros((max_replay_buffer_size, ac_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))

        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.indices_to_replace = np.zeros(max_replay_buffer_size, dtype='uint64')
        self._top = 0
        self._bottom_indexes = 0
        self._top_indexes = 0
        self._size = 0
        self._valid_indeces = []
    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def add_indices_to_replace(self, indices):
        for index in indices:
            self.indices_to_replace[self._top_indexes] = index
            self._top_indexes = (self._top_indexes + 1) % self._max_replay_buffer_size
            self._valid_indeces.remove(index)
        self._size -= len(indices)

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):

        assert not isinstance(self._action_space, Discrete)
        if self._bottom_indexes == self._top_indexes:
            # no invalid samples are present
            index = self._top
        else:
            #replace these samples
            index = self.indices_to_replace[self._bottom_indexes]
        self._observations[index] = observation
        self._actions[index] = action
        self._rewards[index] = reward
        self._terminals[index] = terminal
        self._next_obs[index] = next_observation
        self._valid_indeces.append(index)
        self._advance()

    def _advance(self):
        if self._bottom_indexes == self._top_indexes:
            self._top = (self._top + 1) % self._max_replay_buffer_size
        else:
            self._bottom_indexes = (self._bottom_indexes + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):

        #indices = np.random.randint(0, self._size, batch_size)
        indices = np.random.choice(self._valid_indeces, size=batch_size, replace=False).astype(int)
        try:
            batch = dict(
                observations=self._observations[indices],
                actions=self._actions[indices],
                rewards=self._rewards[indices],
                terminals=self._terminals[indices],
                next_observations=self._next_obs[indices],
            )
        except Exception as e:
            print(e)
        #save indices where to put next batch of observations
        #assumes same number of samples will come and will leave from the buffer
        self.add_indices_to_replace(indices.tolist())
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    def end_epoch(self, epoch):
        return

    def get_snapshot(self):
        return dict(
            _observations=self._observations,
            _next_obs=self._next_obs,
            _actions=self._actions,
            _rewards=self._rewards,
            _terminals=self._terminals,
            _top=self._top,
            _indices_to_replace=self.indices_to_replace,
            _bottom_indexes=self._bottom_indexes,
            _top_indexes=self._top_indexes,
            _valid_indeces=self._valid_indeces,
            _size=self._size,
        )

    def restore_from_snapshot(self, ss):

        for key in ss.keys():
            assert hasattr(self, key)
            setattr(self, key, ss[key])
