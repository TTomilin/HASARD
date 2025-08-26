from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class CostWrapper(gym.Wrapper):
    """
    A wrapper for cost-related functionality.

    This wrapper provides a foundation for implementing cost-related functionality
    while maintaining the standard gymnasium interface. Cost information should
    be handled through the info dictionary returned by the step method.

    By default, this wrapper passes through cost information unchanged. Subclasses
    can override the process_cost method to implement specific cost processing logic.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Execute a step in the environment and process cost information.

        Args:
            action: The action to take in the environment

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                   Cost information is maintained in the info dictionary
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self.process_cost(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def process_cost(self, obs: ObsType, reward: SupportsFloat, terminated: bool,
                     truncated: bool, info: dict[str, Any]) -> dict[str, Any]:
        """
        Process cost information from the environment step.

        By default, this method passes through cost information unchanged.
        Subclasses can override this method to implement specific cost processing logic.

        Args:
            obs: The observation from the environment
            reward: The reward from the environment
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: The info dictionary from the environment

        Returns:
            dict: Updated info dictionary with processed cost information
        """
        return info
