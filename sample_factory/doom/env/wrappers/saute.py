from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict


class Saute(gym.Wrapper):
    """Saute Adapter for Doom.

    Saute is a safe RL algorithm that uses state augmentation to ensure safety. The state
    augmentation is the concatenation of the original state and the safety state. The safety state
    is the safety budget minus the cost divided by the safety budget.

    .. note::
        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    References:
        - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
            David Mguni, Jun Wang, Haitham Bou-Ammar.
        - URL: `Saute <https://arxiv.org/abs/2202.06558>`_

    Args:
        env (Env): The gymnasium environment being wrapped.
        saute_gamma (float): The discount factor for the safety budget calculation.
        unsafe_reward (float): The reward given when the safety state is negative.
        max_ep_len (int): The maximum length of an episode, used in calculating the safety budget.
        num_envs (int): The number of parallel environments. Defaults to 1.
    """

    def __init__(self, env, saute_gamma: float, unsafe_reward: float, max_ep_len: int, num_envs: int = 1):
        super().__init__(env)

        self.safety_budget = (
            self.safety_bound
            * (1 - saute_gamma ** max_ep_len)
            / (1 - saute_gamma)
            / max_ep_len
            * np.ones((num_envs, 1))
        )

        self.num_envs = num_envs
        self.saute_gamma = saute_gamma
        self.unsafe_reward = unsafe_reward
        self._safety_obs = None

        obs_space = self.env.observation_space
        assert isinstance(obs_space, Box), 'Observation space must be Box'
        self.env.observation_space = Dict({
            'obs': Box(low=0, high=255, shape=obs_space.shape, dtype=np.uint8),
            'safety': Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        .. note::
            Additionally, the safety observation will be reset.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._safety_obs = np.ones((self.num_envs, 1), dtype=float)
        return obs, info

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The :meth:`_saute_step` will be called to update the safety observation. Then the reward
            will be updated by :meth:`_safety_reward`.

        Args:
            action: The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        info['original_reward'] = reward
        cost = info.get('cost', 0.0)

        self._safety_step(cost)
        reward = self._safety_reward(reward)

        # autoreset the environment
        done = terminated or truncated
        self._safety_obs = self._safety_obs * (1 - float(done)) + float(done)

        info["episode_extra_stats"] = {
            'original_reward': reward,
            'safety_obs': self._safety_obs,
        }

        return next_obs, reward, terminated, truncated, info

    def _safety_step(self, cost: torch.Tensor) -> None:
        """Update the safety observation.

        Args:
            cost (float): The cost of the current step.
        """
        self._safety_obs -= cost / self.safety_budget
        self._safety_obs /= self.saute_gamma

    def _safety_reward(self, reward: float) -> np.ndarray:
        """Update the reward with the safety observation.

        .. note::
            If the safety observation is greater than 0, the reward will be the original reward.
            Otherwise, the reward will be the unsafe reward.

        Args:
            reward (float): The reward of the current step.

        Returns:
            The final reward determined by the safety observation.
        """
        safe = self._safety_obs > 0
        return np.where(safe, reward, self.unsafe_reward)
