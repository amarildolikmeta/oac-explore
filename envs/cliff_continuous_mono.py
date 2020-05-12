import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CliffWorldContinuousMono(gym.Env):
    def __init__(self, dim=(12,), gamma=0.99, small=-1, large=-5, max_action=1, sigma_noise=0., horizon=np.inf):
        self.horizon = horizon
        self.small = small
        self.large = large
        self.gamma = gamma
        self.dim = dim

        self.min_action = np.array([-max_action])
        self.max_action = np.array([max_action])
        self.min_position = np.array([0])
        self.max_position = np.array([dim[0]])

        self.starting_state_center = np.array([0.5])
        self.goal_state_center = np.array([dim[0] - 0.5])

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
        return np.max(np.abs(new_state - self.goal_state_center)) <= 0.5

    def _is_cliff(self, new_state):
        return 5. <= new_state[0] <= 5.5

    def _generate_initial_state(self):
        retry = True
        while retry:
            new_state = self.starting_state_center + self.sigma_noise * self.np_random.randn(1)
            new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
            retry = self._is_cliff(new_state)
        return new_state

    def step(self, action):
        is_goal = self._is_goal(self.state)
        if is_goal:
            return self.state, 0, True, {}

        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.state + action + self.np_random.randn(1) * self.sigma_noise
        new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)

        is_cliff = self._is_cliff(new_state)
        is_goal = self._is_goal(new_state)

        reward = self.small
        done = False

        if is_cliff:
            new_state = self._generate_initial_state()
            reward = self.large

        if is_goal:
            done = True

        self.state = new_state
        reward /= abs(self.large)
        return self.state, reward, done, {}

    def reset(self):
        self.state = self._generate_initial_state()
        return self.state


if __name__ == '__main__':
    mdp = CliffWorldContinuousMono(sigma_noise=0.)

    s = mdp.reset()
    rets = []
    timesteps = 5000
    count = 0
    n = 1000
    while True:
        t = 0
        ret = 0
        s = mdp.reset()
        while t < 100:
            print("State: ", s)
            a = 1
            s, r, done, _ = mdp.step(a)
            count += 1
            print("Reward: ", r)
            ret += r
            t += 1
            if done:
                print(" Reached Goal!")
                break
            if count > timesteps:
                break
        if count <= timesteps:
            print("Return:", ret)
            rets.append(ret)
        else:
            break
    print("Average Return:", np.mean(rets))
    print("Average error:", np.std(rets) / np.sqrt(len(rets)))
    print("Nume episodes:", len(rets))