import numpy as np
from gym import utils
from envs.mujoco_env import MujocoEnv
import math


diff_to_path = {
    'easy': 'point.xml',
    'medium': 'point_medium.xml',
    'hard': 'point_hard.xml',
    'harder': 'point_harder.xml',
    'maze': 'maze.xml',
    'maze_easy': 'maze_easy.xml'
}


class PointEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, difficulty=None, max_state=500, clip_state=False, terminal=False):
        if difficulty is None:
            difficulty = 'easy'
        model = diff_to_path[difficulty]
        self.max_state = max_state
        self.clip_state = clip_state
        self.bounds = [[0, -9.7, 0], [25, 9.7, 0]]
        self.terminal = terminal
        MujocoEnv.__init__(self, model, 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()

        qpos = next_obs[:3]
        goal = [25.0, 0.0]
        if self.clip_state:
            qvel = next_obs[3:]
            qpos_clipped = np.clip(qpos, a_min=self.bounds[0], a_max=self.bounds[1])
            self.set_state(qpos_clipped, qvel)
            qpos = qpos_clipped
            next_obs = self._get_obs()
        reward = -np.linalg.norm(goal - qpos[:2])
        done = False
        if reward >= -1. and self.terminal:
            done = True
        return next_obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent


if __name__ == "__main__":
    env = PointEnv(difficulty='maze_easy')
    ob = env.reset()
    print(env.action_space)
    done = False
    while not done:
        env.render()
        command = input()
        try:
            x, y = [float(a) for a in command.split(' ')]
        except:
            x, y = 0, 0
        ac = np.array([[x, y]])
        print(ac)
        env.step(ac)
    env.render()