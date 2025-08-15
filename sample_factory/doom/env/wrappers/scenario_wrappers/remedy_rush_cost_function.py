import gymnasium as gym
from vizdoom import GameVariable


class RemedyRushCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        # Initialize for single-agent by default
        self._prev_cost = 0
        self.episode_reward = 0
        # Multi-agent state will be initialized on first step
        self._is_multi_agent = None
        self._prev_cost_multi = None
        self.episode_reward_multi = None

    def reset(self, **kwargs):
        # Reset single-agent state
        self._prev_cost = 0
        self.episode_reward = 0

        # Reset multi-agent state if it exists
        if self._is_multi_agent:
            num_agents = len(self._prev_cost_multi) if self._prev_cost_multi else 0
            self._prev_cost_multi = [0] * num_agents
            self.episode_reward_multi = [0] * num_agents

        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Detect if this is multi-agent environment on first step
        if self._is_multi_agent is None:
            self._is_multi_agent = isinstance(info, list)
            if self._is_multi_agent:
                # Initialize multi-agent state
                num_agents = len(info)
                self._prev_cost_multi = [0] * num_agents
                self.episode_reward_multi = [0] * num_agents

        if self._is_multi_agent:
            # Multi-agent case: info is a list of dicts, reward is a list
            total_episode_reward = 0
            for i in range(len(info)):
                # cost = self.game.get_game_variable(GameVariable.USER1)  # Per agent if needed
                cost = 0
                # goggles = self.game.get_game_variable(GameVariable.USER2)  # Per agent if needed
                goggles = 0
                cost_this_step = cost - self._prev_cost_multi[i]
                info[i]['cost'] = cost_this_step
                self._prev_cost_multi[i] = cost

                self.episode_reward_multi[i] += reward[i]
                total_episode_reward += self.episode_reward_multi[i]

                info[i]['true_objective'] = reward[i]
                info[i]["episode_extra_stats"] = {
                    'cost': cost,
                    'episode_reward': self.episode_reward_multi[i],
                    'agent_id': i,
                    'total_episode_reward': total_episode_reward,
                    'goggles_obtained': goggles,
                }

            # Add total reward logging for multi-agent environments
            # This ensures that total reward is available for logging systems
            for i in range(len(info)):
                info[i]["episode_extra_stats"]['total_episode_reward'] = total_episode_reward
                # Also add individual agent rewards for comparison
                info[i]["episode_extra_stats"]['agent_rewards'] = self.episode_reward_multi.copy()
        else:
            # Single-agent case: info is a dict, reward is a scalar
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
