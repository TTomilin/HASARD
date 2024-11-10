"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""
import json
import os
from os.path import join
from typing import Any, Dict, Tuple, Union

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, RewardWrapper, spaces

from sample_factory.envs.env_utils import num_env_steps
from sample_factory.utils.utils import ensure_dir_exists, log


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


class ResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h, grayscale=True, add_channel_dim=False, area_interpolation=False):
        super(ResizeWrapper, self).__init__(env)

        self.w = w
        self.h = h
        self.grayscale = grayscale
        self.add_channel_dim = add_channel_dim
        self.interpolation = cv2.INTER_AREA if area_interpolation else cv2.INTER_NEAREST

        # Adjust observation space for multi-agent environments
        self.observation_space = self._adjust_observation_space(env.observation_space)

    def _adjust_observation_space(self, old_space):
        if isinstance(old_space, spaces.Tuple):
            # Observation space is a Tuple of spaces for each agent
            new_spaces = tuple(self._calc_new_obs_space(space) for space in old_space.spaces)
            return spaces.Tuple(new_spaces)
        elif isinstance(old_space, spaces.Dict):
            # Observation space is a Dict
            new_spaces = {key: self._calc_new_obs_space(space) for key, space in old_space.spaces.items()}
            return spaces.Dict(new_spaces)
        else:
            # Single observation space
            return self._calc_new_obs_space(old_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]

        if self.grayscale:
            new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]
        else:
            if len(old_space.shape) > 2:
                channels = old_space.shape[-1]
                new_shape = [self.h, self.w, channels]
            else:
                new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]

        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    def _convert_obs(self, obs):
        if obs is None:
            return obs

        obs = cv2.resize(obs, (self.w, self.h), interpolation=self.interpolation)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.add_channel_dim:
            return obs[:, :, None]  # Add new dimension (expected by frameworks like TensorFlow/PyTorch)
        else:
            return obs

    def _observation(self, obs):
        if isinstance(obs, (list, tuple)):
            # Process each observation in the list or tuple
            return type(obs)(self._observation(o) for o in obs)
        elif isinstance(obs, dict):
            # Process each item in the dict
            return {key: self._convert_obs(value) for key, value in obs.items()}
        else:
            # Single observation
            return self._convert_obs(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs), reward, terminated, truncated, info


class RewardScalingWrapper(RewardWrapper):
    def __init__(self, env, scaling_factor):
        super(RewardScalingWrapper, self).__init__(env)
        self._scaling = scaling_factor
        # Adjust reward range if needed
        # self.reward_range = (r * scaling_factor for r in self.reward_range)

    def reward(self, reward):
        if isinstance(reward, (list, tuple)):
            # Scale each reward in the list or tuple
            return type(reward)(self.reward(r) for r in reward)
        else:
            return reward * self._scaling


class TimeLimitWrapper(gym.core.Wrapper):
    def __init__(self, env, limit, random_variation_steps=0):
        super(TimeLimitWrapper, self).__init__(env)
        self._limit = limit
        self._variation_steps = random_variation_steps
        self._num_steps = 0
        self._terminate_in = self._random_limit()

    def _random_limit(self):
        return np.random.randint(-self._variation_steps, self._variation_steps + 1) + self._limit

    def reset(self, **kwargs):
        self._num_steps = 0
        self._terminate_in = self._random_limit()
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation is None:
            return observation, reward, terminated, truncated, info

        self._num_steps += num_env_steps([info])
        if terminated or truncated:
            pass
        elif self._num_steps >= self._terminate_in:
            truncated = True

        return observation, reward, terminated, truncated, info


class PixelFormatChwWrapper(ObservationWrapper):
    """Convert observations from HWC to CHW format."""

    def __init__(self, env):
        super().__init__(env)

        # Adjust observation space for multi-agent environments
        self.observation_space = self._adjust_observation_space(env.observation_space)

    def _adjust_observation_space(self, old_space):
        if isinstance(old_space, spaces.Tuple):
            # Observation space is a Tuple of spaces for each agent
            new_spaces = tuple(self._calc_new_obs_space(space) for space in old_space.spaces)
            return spaces.Tuple(new_spaces)
        elif isinstance(old_space, spaces.Dict):
            # Observation space is a Dict
            new_spaces = {key: self._calc_new_obs_space(space) for key, space in old_space.spaces.items()}
            return spaces.Dict(new_spaces)
        else:
            # Single observation space
            return self._calc_new_obs_space(old_space)

    def _calc_new_obs_space(self, space):
        obs_shape = space.shape
        if len(obs_shape) != 3:
            raise ValueError("Expected observation with 3 dimensions (H, W, C)")

        h, w, c = obs_shape
        new_shape = (c, h, w)
        return spaces.Box(low=space.low.min(), high=space.high.max(), shape=new_shape, dtype=space.dtype)

    def _transpose_obs(self, obs):
        return np.transpose(obs, (2, 0, 1))  # HWC to CHW

    def observation(self, observation):
        if isinstance(observation, (list, tuple)):
            # Process each observation in the list or tuple
            return type(observation)(self.observation(o) for o in observation)
        elif isinstance(observation, dict):
            # Process each item in the dict
            return {key: self._transpose_obs(value) for key, value in observation.items()}
        else:
            # Single observation
            return self._transpose_obs(observation)


class RecordingWrapper(gym.core.Wrapper):
    def __init__(self, env, record_to, player_id):
        super().__init__(env)

        self._record_to = record_to
        self._episode_recording_dirs = []
        self._record_id = 0
        self._frame_id = 0
        self._player_id = player_id
        self._recorded_episode_rewards = []
        self._recorded_episode_shaping_rewards = []
        self._recorded_actions = []

        # Experimental! Recording Doom replay. Does not work in all scenarios, e.g., when there are in-game bots.
        self.unwrapped.record_to = record_to

    def reset(self, **kwargs):
        if self._episode_recording_dirs and self._record_id > 0:
            # Save actions to text file
            for idx, dir in enumerate(self._episode_recording_dirs):
                actions_file_path = join(dir, "actions.json")
                with open(actions_file_path, "w") as actions_file:
                    json.dump(self._recorded_actions[idx], actions_file)

                # Rename previous episode dir
                reward = self._recorded_episode_rewards[idx] + self._recorded_episode_shaping_rewards[idx]
                new_dir_name = dir + f"_r{reward:.2f}"
                os.rename(dir, new_dir_name)
                log.info(
                    "Finished recording %s (reward %.3f, shaping %.3f)",
                    new_dir_name,
                    reward,
                    self._recorded_episode_shaping_rewards[idx],
                )

        self._record_id += 1
        self._frame_id = 0

        # Initialize per-agent recording directories and stats
        num_agents = getattr(self.env, 'num_agents', 1)
        self._episode_recording_dirs = []
        self._recorded_episode_rewards = [0] * num_agents
        self._recorded_episode_shaping_rewards = [0] * num_agents
        self._recorded_actions = [[] for _ in range(num_agents)]

        for idx in range(num_agents):
            dir_name = f"ep_{self._record_id:03d}_agent{idx}"
            episode_dir = join(self._record_to, dir_name)
            ensure_dir_exists(episode_dir)
            self._episode_recording_dirs.append(episode_dir)

        return self.env.reset(**kwargs)

    def _record(self, imgs):
        for idx, img in enumerate(imgs):
            frame_name = f"{self._frame_id:05d}.png"
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(join(self._episode_recording_dirs[idx], frame_name), img)
        self._frame_id += 1

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Record actions and observations for each agent
        if isinstance(action, (list, tuple)):
            for idx, act in enumerate(action):
                self._recorded_actions[idx].append(act)
        else:
            self._recorded_actions[0].append(action)

        if isinstance(obs, (list, tuple)):
            self._record(obs)
        else:
            self._record([obs])

        # Update rewards for each agent
        if isinstance(reward, (list, tuple)):
            for idx, r in enumerate(reward):
                self._recorded_episode_rewards[idx] += r
        else:
            self._recorded_episode_rewards[0] += reward

        # Handle shaping rewards if applicable
        if hasattr(self.env.unwrapped, "_total_shaping_reward"):
            shaping_reward = self.env.unwrapped._total_shaping_reward
            if isinstance(shaping_reward, (list, tuple)):
                for idx, sr in enumerate(shaping_reward):
                    self._recorded_episode_shaping_rewards[idx] = sr
            else:
                self._recorded_episode_shaping_rewards[0] = shaping_reward

        return obs, reward, terminated, truncated, info


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]


# wrapper from CleanRL / Stable Baselines
class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, rew, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# wrapper from CleanRL / Stable Baselines
class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)
    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        # Initialize observation buffers for each agent
        self._obs_buffers = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize observation buffers based on observation shape
        if isinstance(obs, (list, tuple)):
            self._obs_buffers = [np.zeros((2,) + o.shape, dtype=o.dtype) for o in obs]
        else:
            self._obs_buffers = np.zeros((2,) + obs.shape, dtype=obs.dtype)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        info = {}
        terminated = truncated = False
        obs = None

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if isinstance(obs, (list, tuple)):
                for idx, o in enumerate(obs):
                    if i == self._skip - 2:
                        self._obs_buffers[idx][0] = o
                    if i == self._skip - 1:
                        self._obs_buffers[idx][1] = o
            else:
                if i == self._skip - 2:
                    self._obs_buffers[0] = obs
                if i == self._skip - 1:
                    self._obs_buffers[1] = obs
            total_reward += reward
            if terminated or truncated:
                break

        # Max pooling over last observations
        if isinstance(obs, (list, tuple)):
            max_frame = type(obs)(
                np.maximum(buf[0], buf[1]) for buf in self._obs_buffers
            )
        else:
            max_frame = self._obs_buffers.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


# wrapper from CleanRL / Stable Baselines
class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.
    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward:
        :return:
        """
        if isinstance(reward, (list, tuple)):
            # Clip each reward in the list or tuple
            return type(reward)(self.reward(r) for r in reward)
        else:
            return np.sign(reward)


class EpisodeCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.episode_count = 0

    def reset(self, **kwargs) -> Tuple[GymObs, Dict]:
        return self.env.reset(**kwargs)

    def step(self, action: int) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated | truncated:
            extra_stats = info.get("episode_extra_stats", {})
            extra_stats["episode_number"] = self.episode_count
            info["episode_extra_stats"] = extra_stats
            self.episode_count += 1

        return obs, reward, terminated, truncated, info


class ActionCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_counts = None

    def reset(self, **kwargs) -> Tuple[GymObs, Dict]:
        # Reset the action count dictionary at the start of each episode
        self.action_counts = {i: 0 for i in range(self.action_space.n)}
        return self.env.reset(**kwargs)

    def step(self, action: int) -> GymStepReturn:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated | truncated:
            extra_stats = info.get("episode_extra_stats", {})
            extra_stats["episode_number"] = self.episode_count
            info["episode_extra_stats"] = extra_stats
            self.episode_count += 1

        return obs, reward, terminated, truncated, info
