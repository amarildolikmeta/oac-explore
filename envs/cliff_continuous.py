import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CliffWorldContinuous(gym.Env):
    def __init__(self, dim=(12, 4), gamma=0.99, small=-1, large=-100, max_action=1, sigma_noise=0., horizon=np.inf):
        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim

        self.min_action = np.array([-max_action, -max_action])
        self.max_action = np.array([max_action, max_action])
        self.min_position = np.array([0, 0])
        self.max_position = np.array([dim[0], dim[1]])

        self.starting_state_center = np.array([0.5, 0.5])
        self.goal_state_center = np.array([dim[0] - 0.5, 0.5])

        self.sigma_noise = sigma_noise
        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_position, high=self.max_position, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_goal(self, new_state):
        #print(np.max(np.abs(new_state - self.goal_state_center)) <= 0.5)
        return np.max(np.abs(new_state - self.goal_state_center)) <= 0.5

    def _is_cliff(self, new_state):
        return new_state[1] <= 1 and new_state[0] >= 1 and new_state[0] <= self.dim[1] - 1

    def _generare_initial_state(self):
        retry = True
        while retry:
            new_state = self.starting_state_center + self.sigma_noise * self.np_random.randn(2)
            new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
            retry = self._is_cliff(new_state)
        return new_state

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.state + action + self.np_random.randn(2) * self.sigma_noise
        new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)

        is_cliff = self._is_cliff(new_state)
        is_goal = self._is_goal(new_state)

        reward = self.small
        done = False

        if is_cliff:
            new_state = self._generare_initial_state()
            reward = self.large

        if is_goal:
            done = True

        self.state = new_state

        return self.state, reward, done, {}

    def reset(self):
        self.state = self._generare_initial_state()
        return self.state


if __name__ == '__main__':
    mdp = CliffWorldContinuous(sigma_noise=0.1)

    def optimal_policy(s):
        if s[0] >= mdp.dim[0] - 1:
            a = [0, -1]
        elif s[1] >= 1:
            a = [1, 0]
        else:
            a = [0, 1]
        return np.array(a)

    def safe_policy(s):
        if s[0] >= mdp.dim[0] - 1:
            a = [0, -1]
        elif s[1] >= mdp.dim[1] - 1:
            a = [1, 0]
        else:
            a = [0, 1]
        return np.array(a)


    s = mdp.reset()
    rets = []
    n = 1000
    for i in range(n):
        t = 0
        ret = 0
        s = mdp.reset()
        while t < 100:
            a = optimal_policy(s)
            s, r, done, _ = mdp.step(a)
            ret += r
            t += 1
            if done:
                break
        rets.append(ret)
    print("Average Return:", np.mean(rets))
    print("Average error:", np.std(rets) / np.sqrt(n))