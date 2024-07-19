from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import cv2
import gymnasium
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution, GameVariable

from hasard.envs.base import BaseEnv
from hasard.utils.utils import get_screen_resolution


class DoomEnv(BaseEnv):
    """
    A foundational class for creating Doom-based environments in the context of reinforcement learning.

    This class manages the core functionality required for Doom-based environments,
    including game initialization, state management, and rendering.

    Attributes:
        scenario (str): Name of the scenario module.
        frame_skip (int): Number of frames to skip for each action.
        record_every (int): Frequency of recording episodes.
        game (vzd.DoomGame): Instance of the ViZDoom game engine.
        game_res (Tuple[int, int, int]): Resolution of the game screen.
        _action_space (gymnasium.spaces.Discrete): The action space of the environment.
        _observation_space (gymnasium.spaces.Box): The observation space of the environment.
        user_variables (Dict[GameVariable, float]): Custom variables for tracking game state.
        game_variable_buffer (deque): A buffer for storing recent game variables for statistics.

    Args:
        level (int): Level of the task.
        action_space_fn (Callable): Function to generate the action space.
        frame_skip (int): Number of frames to skip for each action.
        record_every (int): Frequency of recording episodes.
        seed (int): Seed for random number generators.
        render (bool): Whether to enable rendering.
        render_sleep (float): Time to sleep between rendering frames.
        test_only (bool): Whether the environment is being used for testing only.
        resolution (str): Predefined resolution for the game screen.
        variable_queue_length (int): Length of the game variable buffer.
    """

    def __init__(self,
                 level: int,
                 constraint: str,
                 frame_skip: int = 4,
                 record_every: int = 100,
                 seed: int = 0,
                 render: bool = True,
                 render_sleep: float = 0.0,
                 test_only: bool = False,
                 resolution: str = None,
                 variable_queue_length: int = 5):
        super().__init__()
        self.level = level
        self.constraint = constraint.lower()
        self.scenario = self.__module__.split('.')[-2]
        self.frame_skip = frame_skip

        # Recording
        self.metadata['render.modes'] = 'rgb_array'
        self.record_every = record_every

        # Determine the directory of the doom scenario
        scenario_dir = f'{Path(__file__).parent.resolve()}/{self.scenario}'

        # Initialize the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/level_{level}_{self.constraint}.wad")
        self.game.set_seed(seed)
        self.render_sleep = render_sleep
        self.render_enabled = render
        self.episode_timeout = self.game.get_episode_timeout()
        if render or test_only:  # Use a higher resolution for watching gameplay
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
            self.frame_skip = 1
        elif resolution:  # Use a particular predefined resolution
            self.game.set_screen_resolution(get_screen_resolution(resolution))
        self.game.init()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self._observation_space = gymnasium.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.available_actions = self.get_available_actions()
        self._action_space = gymnasium.spaces.Discrete(len(self.available_actions))

        # Initialize the user variable dictionary
        self.user_variables = {var: 0.0 for var in self.user_vars}

        # Initialize the game variable queue
        self.game_variable_buffer = deque(maxlen=variable_queue_length)

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env_name}'

    @property
    def user_vars(self) -> List[GameVariable]:
        return []

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return self._observation_space

    @property
    def hard_constraint(self) -> bool:
        return self.constraint == 'hard'

    def get_available_actions(self) -> List[List[bool]]:
        raise NotImplementedError

    def calculate_cost(self) -> float:
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (Optional[int]): Seed for random number generator.
            options (Optional[dict]): Additional options for environment reset.

        Returns:
            observation (np.ndarray): Initial state observation of the environment.
            info (Dict[str, Any]): Additional information about the initial state.
        """
        try:
            self.game.new_episode()
        except vzd.ViZDoomIsNotRunningException:
            print('ViZDoom is not running. Restarting...')
            self.game.init()
            self.game.new_episode()
        self.clear_episode_statistics()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, [1, 2, 0])
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Perform an action in the environment and observe the result.

        Args:
            action (int): An action provided by the agent.

        Returns:
            observation (np.ndarray): The current state observation after taking the action.
            reward (float): The reward achieved by the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (Dict[str, Any]): Additional information about the environment and episode.
        """
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = 0.0
        done = self.game.is_player_dead() or self.game.is_episode_finished() or not state
        truncated = False
        cost, cost_stats = self.calculate_cost()
        info = {
            'cost': cost,
            'cost_stats': cost_stats,
        }

        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.float32(np.zeros(self.game_res))
        if not done:
            self.game_variable_buffer.append(state.game_variables)

        return observation, reward, done, truncated, info

    def render(self, mode="human"):
        """
        Renders the current state of the environment based on the specified mode.

        Args:
            mode (str): The mode for rendering (e.g., 'human', 'rgb_array').

        Returns:
            img (List[np.ndarray] or np.ndarray): Rendered image of the environment state.
        """
        state = self.game.get_state()
        img = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))
        if mode == 'human':
            if not self.render_enabled:
                return [img]
            try:
                # Render the image to the screen with swapped red and blue channels
                cv2.imshow('DOOM', img[:, :, [2, 1, 0]])
                cv2.waitKey(1)
            except Exception as e:
                print(f'Screen rendering unsuccessful: {e}')
                return np.zeros(img.shape)
        return [img]

    def video_schedule(self, episode_id):
        """
        Determines whether a video of the current episode should be recorded.

        Args:
            episode_id (int): The identifier of the current episode.

        Returns:
            bool: True if the episode should be recorded, False otherwise.
        """
        return not episode_id % self.record_every

    def clear_episode_statistics(self) -> None:
        """
        Clears or resets statistics collected during an episode.
        """
        self.user_variables.fromkeys(self.user_variables, 0.0)
        self.game_variable_buffer.clear()

    def close(self):
        """
        Performs cleanup and closes the environment.
        """
        self.game.close()
