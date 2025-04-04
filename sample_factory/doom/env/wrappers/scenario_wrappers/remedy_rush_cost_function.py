import gymnasium as gym
from vizdoom import GameVariable


class RemedyRushCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_cost = 0
        self.episode_reward = 0

    def reset(self, **kwargs):
        self._prev_cost = 0
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        cost = self.game.get_game_variable(GameVariable.USER1)
        goggles = self.game.get_game_variable(GameVariable.USER2)
        cost_this_step = cost - self._prev_cost
        info['cost'] = cost_this_step
        self._prev_cost = cost

        self.episode_reward += reward

        info['true_objective'] = reward
        info["episode_extra_stats"] = {
            'cost': cost,
            'episode_reward': self.episode_reward,
            'goggles_obtained': goggles,
        }

        return observation, reward, terminated, truncated, info
