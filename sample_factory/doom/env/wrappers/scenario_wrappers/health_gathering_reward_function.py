import gymnasium as gym


class HealthGatheringRewardFunction(gym.Wrapper):

    def __init__(self, env, health_reward=1.0, death_penalty=-10.0):
        super().__init__(env)
        self.health_reward = health_reward
        self.death_penalty = death_penalty
        self._prev_health = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._prev_health = [info['HEALTH'] for info in infos]
        return observations, infos

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = self.env.step(action)

        for i, info in enumerate(infos):
            cur_h = info.get("HEALTH", self._prev_health[i])
            prev_h = self._prev_health[i]

            if cur_h > prev_h:
                rewards[i] += self.health_reward
            if info.get("just_died", False):
                rewards[i] += self.death_penalty

            self._prev_health[i] = cur_h
        return observations, rewards, terminateds, truncateds, infos
