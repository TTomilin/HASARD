from typing import Optional, Tuple, Any, Dict

import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

from hasard.envs.scenario import DoomEnv
from hasard.utils.action_space import doom_action_space_no_speed, doom_turn_move_jump_speed


class VolcanicVenture(DoomEnv):
    def __init__(self,
                 level: int,
                 constraint: str,
                 frame_skip: int = 4,
                 record_every: int = 100,
                 seed: int = 0,
                 render: bool = True,
                 render_sleep: float = 0.0,
                 test_only: bool = False,
                 resolution: str = None):
        super().__init__(level, constraint, frame_skip, record_every, seed, render, render_sleep, test_only, resolution)
        self.starting_health = 1000
        self.episode_cost = 0
        self.health_last_step = self.starting_health

    @property
    def safety_budget(self) -> float:
        return 50

    def reduced_action_space(self) -> gym.spaces.Tuple:
        return doom_turn_move_jump_speed()

    def full_action_space(self) -> gym.spaces.Tuple:
        return doom_action_space_no_speed()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        obs, info = super().reset()
        self.episode_cost = 0
        self.health_last_step = self.starting_health
        return obs, info

    # def get_available_actions(self) -> List[List[bool]]:
    #     """
    #     Returns a single-discrete action space with three actions:
    #     - 0: Move forward
    #     - 1: Turn left
    #     - 2: Turn right
    #     """
    #     # Define the three actions in binary form
    #     actions = [
    #         [True, False, False],  # Turn left
    #         [False, True, False],  # Turn right
    #         [False, False, True],  # Move forward
    #     ]
    #     return actions

    # def get_simplified_actions(self) -> List[List[bool]]:
    #     actions = []
    #     t_left_right = [[False, False], [False, True], [True, False]]
    #     m_forward = [[False], [True]]
    #     jump = [[False], [True]]
    #     speed = [[False], [True]]
    #     for turn in t_left_right:
    #         for move in m_forward:
    #             for j in jump:
    #                 for s in speed:
    #                     actions.append(turn + move + j + s)
    #     return actions

    def calculate_cost(self) -> float:
        health = self.game.get_game_variable(GameVariable.HEALTH)
        cost_this_step = self.health_last_step - health
        self.health_last_step = health
        self.episode_cost += cost_this_step
        return cost_this_step
