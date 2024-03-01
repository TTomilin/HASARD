import gymnasium as gym

STARTING_Z_COORD = 0


class SpelunkingRewardFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_z = STARTING_Z_COORD

    def reset(self, **kwargs):
        self._prev_z = STARTING_Z_COORD
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        pos_z = info['POSITION_Z']
        reward = self._prev_z - pos_z
        self._prev_z = pos_z
        return observation, reward, terminated, truncated, info
