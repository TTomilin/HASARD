import gymnasium as gym
from vizdoom import GameVariable


class VolcanicVentureRewardFunction(gym.Wrapper):

    def __init__(self, env, reward_scaler: float = 1.0, penalty_scaler: float = 0.1):
        super().__init__(env)
        self.reward_scaler = reward_scaler
        self.penalty_scaler = penalty_scaler
        self._prev_health = None
        self._start_health = None
        self._prev_armor = None
        self._start_armor = None
        self._episode_rewards = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        n_agents = len(infos)
        self._start_health = self._prev_health = [1000] * n_agents
        self._start_armor = self._prev_armor = [0] * n_agents
        self._episode_rewards = [0.0] * n_agents
        return observations, infos

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = self.env.step(action)

        for i, info in enumerate(infos):
            prev_h = self._prev_health[i]
            prev_a = self._prev_armor[i]
            cur_h = info.get("HEALTH", prev_h)
            cur_a = info.get("ARMOR", prev_a)

            # Health-loss penalty
            health_drop = max(prev_h - cur_h, 0.0)
            penalty_this_step = health_drop * self.penalty_scaler

            # Armor-gain reward
            armor_gain = cur_a - prev_a
            reward_this_step = armor_gain * self.reward_scaler

            # rewards[i] = reward_this_step - penalty_this_step  # TODO restore this if we know that training works fine
            rewards[i] = reward_this_step

            # Per-step bookkeeping
            info['health_loss'] = health_drop
            info['armor_gain'] = armor_gain
            self._episode_rewards[i] += reward_this_step

            info['true_objective'] = reward_this_step
            if terminateds[i] or truncateds[i]:
                extra = info.get("episode_extra_stats", {})
                extra.update({
                    "episode_reward": self._episode_rewards[i],   # armor-gain shaped sum
                })
                info["episode_extra_stats"] = extra

            # Update prevs
            self._prev_health[i] = cur_h
            self._prev_armor[i] = cur_a

        return observations, rewards, terminateds, truncateds, infos
