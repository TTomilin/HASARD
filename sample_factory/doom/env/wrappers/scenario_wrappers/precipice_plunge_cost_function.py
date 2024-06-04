import gymnasium as gym
from vizdoom import GameVariable

STARTING_HEALTH = 500
COST_SCALER = 0.1


class PrecipicePlungeCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_health = STARTING_HEALTH
        self._episode_reward = 0

    def reset(self, **kwargs):
        self._prev_health = STARTING_HEALTH
        self._episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        health = info['HEALTH']
        # The cost for the hard constraint will come from the environment
        cost = cost_this_step = self.game.get_game_variable(GameVariable.USER1)
        if cost == 0:
            # Calculate the cost for the soft constraint based on how much health was lost
            cost = (STARTING_HEALTH - health) * COST_SCALER
            cost_this_step = (self._prev_health - health) * COST_SCALER

        self._prev_health = health
        self._episode_reward += reward

        info['cost'] = cost_this_step
        info['true_objective'] = reward
        info["episode_extra_stats"] = {
            'cost': cost,
            'episode_reward': self._episode_reward,
        }

        return observation, reward, terminated, truncated, info
