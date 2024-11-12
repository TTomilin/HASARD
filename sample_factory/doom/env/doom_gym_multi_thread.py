import math
import time
from queue import Queue
from threading import Thread
from typing import Optional, Dict, Tuple, Any

import cv2
import gym
import gymnasium as gym
import numpy as np
import pygame
import vizdoom as vzd
from vizdoom import ScreenResolution, Mode

from sample_factory.doom.env.doom_gym import VizdoomEnv

resolutions = {'1920x1080': ScreenResolution.RES_1920X1080,
               '1600x1200': ScreenResolution.RES_1600X1200,
               '1280x720': ScreenResolution.RES_1280X720,
               '800x600': ScreenResolution.RES_800X600,
               '640x480': ScreenResolution.RES_640X480,
               '320x240': ScreenResolution.RES_320X240,
               '160x120': ScreenResolution.RES_160X120}


def get_screen_resolution(resolution: str) -> ScreenResolution:
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}')
    return resolutions[resolution]


class VizdoomMultiAgentEnv(VizdoomEnv):
    def __init__(
            self,
            config_file: str,
            action_space: gym.Space,
            safety_bound: float,
            unsafe_reward: float,
            timeout: int,
            level=1,
            constraint='soft',
            coord_limits=None,
            max_histogram_length=None,
            show_automap=False,
            skip_frames=1,
            async_mode=False,
            record_to=None,
            resolution: str = None,
            seed: Optional[int] = None,
            render_mode: Optional[str] = None,
            num_agents: int = 2,
            host_address: str = "127.0.0.1",
            port: int = 5029,
    ):
        super().__init__(config_file, action_space, safety_bound, unsafe_reward, timeout, level, constraint,
                         coord_limits, max_histogram_length, show_automap, skip_frames, async_mode, record_to,
                         resolution, seed, render_mode)
        self.num_agents = num_agents
        self.host_address = host_address
        self.port = port

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )

        # Create command queues and threads for each agent
        self.command_queues = [Queue() for _ in range(self.num_agents)]
        self.result_queues = [Queue() for _ in range(self.num_agents)]
        self.threads = []

        for i in range(self.num_agents):
            is_host = i == 0
            thread = Thread(
                target=self._game_thread,
                args=(self.command_queues[i], self.result_queues[i], i, is_host),
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

        time.sleep(2.0)  # Allow time for all threads to initialize

    def _game_thread(self, command_queue, result_queue, instance_id, is_host):
        import cProfile

        pr = cProfile.Profile()
        pr.enable()

        # Each game instance runs in its own thread
        game = vzd.DoomGame()
        game.load_config(self.config_path)

        # Disable rendering and sound
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_screen_resolution(get_screen_resolution(self.resolution))
        game.set_mode(Mode.ASYNC_PLAYER)  # Use ASYNC_PLAYER for multiplayer
        game.set_ticrate(10000)

        # Multiplayer settings
        if is_host:
            game.add_game_args(
                f"-host {self.num_agents} -port {self.port} -netmode 1 +timelimit 10.0 +sv_spawnfarthest 1")
            game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
        else:
            game.add_game_args(f"-join {self.host_address} -port {self.port} -netmode 1")
            game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")

        game.init()

        while True:
            cmd, data = command_queue.get()
            if cmd == 'reset':
                game.new_episode()
                observation = self._get_observation(game)
                result_queue.put(('reset', observation))
            elif cmd == 'step':
                action = data
                observation, reward, done, info = self._execute_action(game, action)
                result_queue.put(('step', (observation, reward, done, info)))
            elif cmd == 'render':
                observation = self._get_observation(game)
                result_queue.put(('render', observation))
            elif cmd == 'close':
                game.close()
                break
            else:
                print(f"Unknown command: {cmd}")

        pr.disable()
        profile_filename = f'game_process_{instance_id}_profile.prof'
        pr.dump_stats(profile_filename)

    def _get_observation(self, game):
        state = game.get_state()
        if state and state.screen_buffer is not None:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
        else:
            observation = np.zeros((240, 320, 3), dtype=np.uint8)
        return observation

    def _execute_action(self, game, action):
        if not game.is_episode_finished():
            game.make_action(action, self.skip_frames)
            observation = self._get_observation(game)
            reward = game.get_last_reward()
            done = game.is_episode_finished()
            info = {}
        else:
            observation = self._get_observation(game)
            reward = 0.0
            done = True
            info = {}
        return observation, reward, done, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])
        for command_queue in self.command_queues:
            command_queue.put(('reset', None))

        observations = []
        for result_queue in self.result_queues:
            cmd, observation = result_queue.get()
            assert cmd == 'reset'
            observations.append(observation)
        return observations, {}

    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        for command_queue, action in zip(self.command_queues, actions):
            command_queue.put(('step', action))

        observations = []
        rewards = []
        dones = []
        infos = []
        for result_queue in self.result_queues:
            cmd, data = result_queue.get()
            assert cmd == 'step'
            observation, reward, done, info = data
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        truncated = [False] * len(dones)  # TODO implement proper truncation
        return observations, rewards, dones, truncated, infos

    def close(self):
        for command_queue in self.command_queues:
            command_queue.put(('close', None))
        for thread in self.threads:
            thread.join()

    def render(self) -> Optional[list]:
        if not self.render_mode:
            return None

        for command_queue in self.command_queues:
            command_queue.put(('render', None))

        frames = []
        for result_queue in self.result_queues:
            cmd, frame = result_queue.get()
            assert cmd == 'render'
            frames.append(frame)

        max_screen_width, max_screen_height = 1920, 1080
        if self.render_mode == "human":
            num_agents = len(frames)
            frame_h, frame_w = frames[0].shape[:2]
            columns = min(num_agents, max_screen_width // frame_w)
            rows = math.ceil(num_agents / columns)
            scale = min(max_screen_width / (columns * frame_w), max_screen_height / (rows * frame_h), 1.0)

            display_w, display_h = int(columns * frame_w * scale), int(rows * frame_h * scale)
            if not self.screen or self.screen.get_size() != (display_w, display_h):
                pygame.init()
                self.screen = pygame.display.set_mode((display_w, display_h))
                pygame.display.set_caption("Multi-Agent Environment")

            for idx, frame in enumerate(frames):
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
                x, y = (idx % columns) * int(frame_w * scale), (idx // columns) * int(frame_h * scale)
                surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surface, (x, y))

            pygame.display.flip()
            time.sleep(0.01)
        elif self.render_mode == "rgb_array":
            return frames
