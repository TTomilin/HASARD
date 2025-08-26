from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution

from hasard.utils.rendering import segment_and_draw_boxes
from hasard.utils.utils import get_screen_resolution
from sample_factory.algo.utils.spaces.discretized import Discretized


class DoomEnv(gym.Env, ABC):
    """
    A foundational class for creating Doom-based environments in the context of reinforcement learning.

    This class manages the core functionality required for Doom-based environments,
    including game initialization, state management, action processing, and rendering.

    Attributes:
        scenario_name (str): Name of the scenario module.
        frame_skip (int): Number of frames to skip for each action.
        record_every (int): Frequency of recording episodes.
        game (vzd.DoomGame): Instance of the ViZDoom game engine.
        game_res (Tuple[int, int, int]): Resolution of the game screen (height, width, channels).
        observation_space (gym.spaces.Box): The observation space of the environment.
        action_space (gym.spaces.Space): The action space of the environment.
        composite_action_space (bool): Indicates if the action space is composite (has multiple subspaces).
        delta_actions_scaling_factor (float): Scaling factor for continuous delta actions.
    """

    def __init__(self,
                 level: int = 1,
                 seed: int = 0,
                 frame_skip: int = 4,
                 record_every: int = 100,
                 render_mode: str = 'human',
                 render_boxes: bool = False,
                 render_segmented: bool = False,
                 full_actions: bool = False,
                 hard_constraint: bool = False,
                 resolution: Optional[str] = None,
                 safety_budget: Optional[float] = None):
        """
        Initializes the Doom environment.

        Args:
            level (int, optional): Level of the task. Defaults to 1.
            constraint (str, optional): Constraint type ('soft' or 'hard'). Defaults to 'soft'.
            seed (int, optional): Seed for random number generators. Defaults to 0.
            frame_skip (int, optional): Number of frames to skip for each action. Defaults to 4.
            record_every (int, optional): Frequency of recording episodes. Defaults to 100.
            render_mode (str, optional): Mode to render the environment ('human' or 'rgb_array'). Defaults to 'rgb_array'.
            full_actions (bool, optional): Whether to use the full action space. Defaults to False.
            resolution (str, optional): Predefined resolution for the game screen. Defaults to None.
            safety_budget (float, optional): The safety budget for the environment. If None, subclasses must provide their own default. Defaults to None.
        """
        super().__init__()
        self.level = level
        self.hard_constraint = hard_constraint
        self.scenario_name = self.__module__.split('.')[-2]
        self.frame_skip = frame_skip
        self._safety_budget = safety_budget

        # Initialize metadata properly
        self.metadata = {'render_modes': ['rgb_array', 'human']}
        self.record_every = record_every

        # Determine the directory of the Hasard scenario
        constraint = '_hard' if hard_constraint else ''
        scenario_dir = f'{Path(__file__).parent.resolve()}/{self.scenario_name}'
        scenario_path = f"{scenario_dir}/level_{level}{constraint}.wad"
        config_path = f"{scenario_dir}/conf.cfg"

        # Initialize the Doom game instance
        self.game = vzd.DoomGame()
        self.game.set_seed(seed)
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.episode_timeout = self.game.get_episode_timeout()

        # Set the available buttons based on the action space
        if full_actions:
            actions = self.full_actions()
            self.game.set_available_buttons(actions)

        # Set screen resolution based on render mode or user-defined resolution
        if render_mode == 'human':  # Use a higher resolution for watching gameplay
            self.game.set_window_visible(True)
            self.game.set_screen_resolution(ScreenResolution.RES_1280X1024)
            self.frame_skip = 1
        elif resolution:  # User-defined resolution
            self.game.set_screen_resolution(get_screen_resolution(resolution))
        self.game.init()

        # Rendering options
        self.render_segmented = render_segmented
        self.render_boxes = render_boxes

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.action_space = self.full_action_space() if full_actions else self.reduced_action_space()
        self.composite_action_space = hasattr(self.action_space, "spaces")
        self.delta_actions_scaling_factor = 7.5

    @property
    def safety_budget(self) -> float:
        """
        Retrieves the safety budget for the environment.

        Returns:
            float: The safety budget.
        """
        if self._safety_budget is not None:
            return self._safety_budget
        elif hasattr(self, '_default_safety_budget'):
            return self._default_safety_budget()
        else:
            raise NotImplementedError("safety_budget must be provided either as a parameter or overridden by subclass")

    @abstractmethod
    def reduced_action_space(self) -> gym.spaces.Space:
        """
        Defines a simplified action space for the environment.

        Returns:
            gym.spaces.Space: The simplified action space.
        """
        pass

    @abstractmethod
    def full_action_space(self) -> gym.spaces.Space:
        """
        Defines the full action space for the environment using a predefined action space function.

        Returns:
            gym.spaces.Space: The full action space.
        """
        pass

    @abstractmethod
    def full_actions(self) -> List[str]:
        """
        Defines the list of full actions.

        Returns:
            gym.spaces.Tuple: The full action space.
        """
        pass

    @abstractmethod
    def calculate_cost(self) -> float:
        """
        Calculates the cost associated with the current state.

        Returns:
            float: The calculated cost.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the environment.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        return {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to its initial state and returns the initial observation.

        Args:
            seed (Optional[int], optional): Seed for the random number generator. Defaults to None.
            options (Optional[dict], optional): Additional options for environment reset. Defaults to None.

        Raises:
            vzd.ViZDoomIsNotRunningException: If ViZDoom is not running and cannot be restarted.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - observation (np.ndarray): Initial state observation of the environment.
                - info (Dict[str, Any]): Additional information about the initial state.
        """
        # Call parent reset to handle seed properly
        super().reset(seed=seed)

        # Set game seed if provided
        if seed is not None:
            self.game.set_seed(seed)

        try:
            self.game.new_episode()
        except vzd.ViZDoomIsNotRunningException:
            print('ViZDoom is not running. Restarting...')
            self.game.init()
            self.game.new_episode()

        state = self.game.get_state().screen_buffer
        state = np.transpose(state, [1, 2, 0])
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes an action in the environment and returns the resulting state.

        Args:
            action (int): An action provided by the agent, typically an index representing a specific action.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                - observation (np.ndarray): The current state observation after taking the action.
                - reward (float): The reward achieved by the action.
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Whether the episode was truncated.
                - info (Dict[str, Any]): Additional information about the environment and episode.
        """
        # Convert the action index to the actual Doom action
        doom_action = self._convert_actions(action)
        self.game.set_action(doom_action)
        self.game.advance_action(self.frame_skip)
        reward = self.game.get_last_reward()
        state = self.game.get_state()
        terminated = self.game.is_player_dead() or self.game.is_episode_finished() or not state
        truncated = self.game.get_episode_time() >= self.episode_timeout
        cost = self.calculate_cost()
        stats = self.get_statistics()
        info = {
            'cost': cost,
            'env_stats': stats,
        }

        # Process the new state
        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.float32(np.zeros(self.game_res))
        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Renders the current state of the environment based on the specified mode.

        Args:
            mode (str, optional): The mode for rendering. Options are:
                - 'human': Render to the screen using OpenCV.
                - 'rgb_array': Return an RGB array of the current frame.
                Defaults to "human".

        Returns:
            Optional[np.ndarray]:
                - For 'rgb_array' mode, returns the current frame as an RGB array.
                - For 'human' mode, returns None.
                - If rendering fails, returns a black image array.
        """
        state = self.game.get_state()
        screen = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))

        if mode == 'human':
            try:
                # Render the screen with swapped red and blue channels
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                # cv2.imshow('HASARD', screen)

                # Render a screen with segmented objects and bounding boxes
                if state and state.labels_buffer is not None:
                    segmented_obs = segment_and_draw_boxes(screen, state, do_segment=self.render_segmented,
                                                           do_boxes=self.render_boxes)
                    cv2.imshow('Objects', segmented_obs)

                # Render a screen for the depth buffer
                if state and state.depth_buffer is not None:
                    depth_buffer = state.depth_buffer
                    cv2.imshow("Depth", depth_buffer)

                    # Normalize depth buffer for better visualization
                    # normalized_depth = (255 * (depth_buffer / np.max(depth_buffer))).astype(np.uint8)
                    # cv2.imshow("Normalized Depth Buffer", normalized_depth)

                cv2.waitKey(1)
            except Exception as e:
                print(f'Screen rendering unsuccessful: {e}')
                return np.zeros(screen.shape)
            return screen
        elif mode == 'rgb_array':
            return screen
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def video_schedule(self, episode_id: int) -> bool:
        """
        Determines whether a video of the current episode should be recorded.

        Args:
            episode_id (int): The identifier of the current episode.

        Returns:
            bool: True if the episode should be recorded, False otherwise.
        """
        return not episode_id % self.record_every

    def close(self):
        """
        Performs cleanup and closes the environment.

        Closes the ViZDoom game instance and releases any associated resources.
        """
        self.game.close()

    def _convert_actions(self, actions: int) -> list:
        """
        Converts an action index from the gym action space to the action list expected by ViZDoom.

        Handles both composite and simple action spaces, converting each sub-action appropriately.

        Args:
            actions (int): The action index provided by the agent.

        Raises:
            NotImplementedError: If the action subspace type is not supported.

        Returns:
            list: A flattened list of actions compatible with ViZDoom.
        """
        if self.composite_action_space:
            # Composite action space with multiple subspaces
            spaces = self.action_space.spaces
        else:
            # Simple action space, treat it as a composite of length 1
            spaces = (self.action_space,)
            actions = (actions,)

        actions_flattened = []
        for i, action in enumerate(actions):
            if isinstance(spaces[i], Discretized):
                # Discretized continuous action
                # Convert discretized action to continuous value
                continuous_action = spaces[i].to_continuous(action)
                actions_flattened.append(continuous_action)
            elif isinstance(spaces[i], gym.spaces.Discrete):
                # Standard discrete action
                num_non_idle_actions = spaces[i].n - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action is a no-op
                actions_flattened.extend(action_one_hot.tolist())
            elif isinstance(spaces[i], gym.spaces.Box):
                # Continuous action
                actions_flattened.extend((action * self.delta_actions_scaling_factor).tolist())
            else:
                raise NotImplementedError(f"Action subspace type {type(spaces[i])} is not supported!")

        return actions_flattened
