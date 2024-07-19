from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class CostWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, SupportsFloat, bool, bool, dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        cost = info.get('cost', 0)
        return state, reward, cost, done, truncated, info
