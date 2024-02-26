import gymnasium as gym
from vizdoom import GameVariable


STARTING_HEALTH = 1000


class VolcanicVentureCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_health = STARTING_HEALTH

    def reset(self, **kwargs):
        self._prev_health = STARTING_HEALTH
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        health = self.game.get_game_variable(GameVariable.HEALTH)
        total_cost = STARTING_HEALTH - health
        cost_this_step = self._prev_health - health
        info['cost'] = cost_this_step
        info['total_cost'] = total_cost
        self._prev_health = health

        info["episode_extra_stats"] = {'cost': total_cost}

        return observation, reward, terminated, truncated, info
