import gymnasium as gym

STARTING_HEALTH = 1000
COST_SCALER = 0.1


class SpelunkingCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_health = STARTING_HEALTH

    def reset(self, **kwargs):
        self._prev_health = STARTING_HEALTH
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        health = info['HEALTH']
        total_cost = (STARTING_HEALTH - health) * COST_SCALER
        cost_this_step = (self._prev_health - health) * COST_SCALER
        info['cost'] = cost_this_step
        self._prev_health = health

        info["episode_extra_stats"] = {
            'cost': total_cost,
            'original_reward': reward,
        }

        return observation, reward, terminated, truncated, info
