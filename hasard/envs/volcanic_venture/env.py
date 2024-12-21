from typing import Optional, Tuple, Any, Dict, List

import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

from hasard.envs.scenario import DoomEnv
from hasard.utils.action_space import doom_action_space_no_speed, doom_turn_move_jump_speed, doom_actions_full, \
    doom_action_space_full


class VolcanicVenture(DoomEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.starting_health = 1000
        self.episode_cost = 0
        self.health_last_step = self.starting_health

    @property
    def safety_budget(self) -> float:
        return 50

    def reduced_action_space(self) -> gym.spaces.Tuple:
        return doom_turn_move_jump_speed()

    def full_action_space(self) -> gym.spaces.Tuple:
        return doom_action_space_full()

    def full_actions(self) -> List[str]:
        return doom_actions_full()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        obs, info = super().reset()
        self.episode_cost = 0
        self.health_last_step = self.starting_health
        return obs, info

    def calculate_cost(self) -> float:
        health = self.game.get_game_variable(GameVariable.HEALTH)
        cost = self.health_last_step - health
        self.health_last_step = health
        self.episode_cost += cost
        return cost

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats['episode_cost'] = self.episode_cost
        return stats
