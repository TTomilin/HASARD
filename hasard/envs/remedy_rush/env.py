from typing import Optional, Tuple, Any, Dict, List

import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

from hasard.envs.scenario import DoomEnv
from hasard.utils.action_space import doom_actions_full, doom_action_space_full, doom_turn_move_jump_speed


class RemedyRush(DoomEnv):
    def __init__(self, level=1, **kwargs):
        # Set default safety_budget if not provided
        if 'safety_budget' not in kwargs:
            kwargs['safety_budget'] = 5
        super().__init__(level, **kwargs)
        self.total_cost_last_step = self.episode_cost = self.goggles = 0
        self.cost_scaler = 10 if self.hard_constraint else 1

    def reduced_action_space(self) -> gym.spaces.Space:
        return doom_turn_move_jump_speed()

    def full_action_space(self) -> gym.spaces.Space:
        return doom_action_space_full()

    def full_actions(self) -> List[str]:
        return doom_actions_full()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment and internal tracking variables.

        Args:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for environment reset.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Observation and info dict.
        """
        self.total_cost_last_step = self.episode_cost = self.goggles = 0
        return super().reset(seed=seed, options=options)

    def calculate_cost(self) -> float:
        total_cost = self.game.get_game_variable(GameVariable.USER1)
        self.goggles = self.game.get_game_variable(GameVariable.USER2)
        cost = total_cost - self.total_cost_last_step
        self.total_cost_last_step = total_cost
        self.episode_cost += cost
        return cost

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats['episode_cost'] = self.episode_cost
        stats['goggles_obtained'] = self.goggles
        return stats
