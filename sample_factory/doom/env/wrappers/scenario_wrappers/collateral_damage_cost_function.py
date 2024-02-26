import gymnasium as gym
from vizdoom import GameVariable


class DoomCollateralDamageCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_cost = 0

    def reset(self, **kwargs):
        self._prev_cost = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        cost = self.game.get_game_variable(GameVariable.USER1)
        cost_this_step = cost - self._prev_cost
        info['cost'] = cost_this_step
        info['total_cost'] = cost
        if cost > self._prev_cost:
            self._prev_cost = cost

        info["episode_extra_stats"] = {'cost': cost}

        return observation, reward, terminated, truncated, info
