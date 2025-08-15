import math
import time
from multiprocessing import Process, Pipe, shared_memory
from typing import Optional, Dict, Tuple, Any

import cv2
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


def game_process(config_path, resolution, skip_frames, pipe, instance_id, is_host, num_agents, port, shm_name,
                 obs_shape):
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

    # Each game instance runs in its own process
    game = vzd.DoomGame()
    game.load_config(config_path)

    # Disable rendering and sound
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_mode(Mode.ASYNC_PLAYER)  # Use ASYNC_PLAYER for multiplayer
    game.set_ticrate(1000)

    # Multiplayer settings
    if is_host:
        # Host game instance
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode 1 +timelimit 10.0 +sv_spawnfarthest 1"
        )
        game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
        game.add_game_args(f"+playernumber {instance_id}")
    else:
        # Join game instance
        game.add_game_args(f"-join 127.0.0.1 -port {port} -netmode 1")
        game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
        game.add_game_args(f"+playernumber {instance_id}")

    game.init()

    # Connect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    observations = np.ndarray(obs_shape, dtype=np.uint8, buffer=existing_shm.buf)

    try:

        # Main loop
        while True:
            cmd, data = pipe.recv()

            if cmd == 'step':
                action = data
                if not game.is_episode_finished():
                    game.make_action(action, skip_frames)
                    state = game.get_state()
                    reward = game.get_last_reward()
                    done = game.is_episode_finished()

                    if state and state.screen_buffer is not None:
                        observation = np.transpose(state.screen_buffer, (1, 2, 0))
                    else:
                        observation = np.zeros(obs_shape[1:], dtype=np.uint8)

                    info = {}
                else:
                    reward = 0.0
                    done = True
                    observation = np.zeros(obs_shape[1:], dtype=np.uint8)
                    info = {}

                # Write observation into shared memory
                observations[instance_id] = observation

                # Send minimal data back via the pipe
                pipe.send((reward, done, info))

            elif cmd == 'reset':
                # Reset the environment
                game.new_episode()
                # Get initial state
                state = game.get_state()
                if state and state.screen_buffer is not None:
                    observation = np.transpose(state.screen_buffer, (1, 2, 0))
                else:
                    observation = np.zeros(obs_shape[1:], dtype=np.uint8)
                # Write observation into shared memory
                observations[instance_id] = observation
                # Signal that reset is done
                pipe.send(None)

            elif cmd == 'render':
                # Get current screen buffer for rendering
                state = game.get_state()
                if state and state.screen_buffer is not None:
                    frame = np.transpose(state.screen_buffer, (1, 2, 0))
                    pipe.send(frame)
                else:
                    # Send a black frame if no state available
                    pipe.send(np.zeros(obs_shape[1:], dtype=np.uint8))

            elif cmd == 'close':
                game.close()
                pipe.close()
                # Close the shared memory in the agent process
                existing_shm.close()
                break
    finally:
        pr.disable()
        profile_filename = f'game_process_{instance_id}_profile.prof'
        pr.dump_stats(profile_filename)


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
            use_depth_buffer=False,
            render_depth_buffer=False,
            render_with_bounding_boxes=False,
            segment_objects=False,
            skip_frames=1,
            async_mode=False,
            record_to=None,
            env_modification: str = None,
            resolution: str = "160x120",
            seed: Optional[int] = None,
            render_mode: Optional[str] = None,
            num_agents: int = 2,
            host_address: str = "127.0.0.1",
            port: int = 5029,
    ):
        super().__init__(config_file, action_space, safety_bound, unsafe_reward, timeout, level, constraint,
                         coord_limits, max_histogram_length, show_automap, use_depth_buffer, render_depth_buffer,
                         render_with_bounding_boxes, segment_objects, skip_frames, async_mode, record_to,
                         env_modification, resolution, seed, render_mode)
        self.num_agents = num_agents
        self.host_address = host_address
        self.port = port
        self.resolution = resolution

        # Initialize pygame screen for rendering
        self.screen = None

        parts = resolution.lower().split("x")
        width = int(parts[0])
        height = int(parts[1])

        # Define observation and action spaces
        obs_shape = (height, width, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Create shared memory for observations
        multi_obs_shape = (self.num_agents, height, width, 3)
        obs_size = np.prod(multi_obs_shape) * np.dtype(np.uint8).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=obs_size)
        self.observations = np.ndarray(multi_obs_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Create pipes and processes
        self.parent_pipes = []
        self.processes = []

        for i in range(self.num_agents):
            parent_conn, child_conn = Pipe()
            self.parent_pipes.append(parent_conn)

            is_host = i == 0

            # Create and start the game process, passing shared memory details
            process = Process(
                target=game_process,
                args=(
                    self.config_path, self.resolution, self.skip_frames, child_conn,
                    i, is_host, self.num_agents, self.port,
                    self.shm.name, multi_obs_shape
                ),
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

        # Wait a bit to ensure all processes are ready
        time.sleep(1.0)

    # @profile
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])

        for pipe in self.parent_pipes:
            pipe.send(('reset', None))

        # Wait for all agents to acknowledge reset
        for pipe in self.parent_pipes:
            pipe.recv()

        observations = [self.observations[i] for i in range(self.num_agents)]

        return observations, {}

    # @profile
    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))

        rewards = []
        dones = []
        infos = []

        for pipe in self.parent_pipes:
            reward, done, info = pipe.recv()
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        observations = [self.observations[i] for i in range(self.num_agents)]

        truncated = [False] * len(dones)  # Implement proper truncation if needed

        return observations, rewards, dones, truncated, infos

    def close(self):
        # Prevent multiple close calls
        if hasattr(self, '_closed') and self._closed:
            return

        # Send close messages to all processes, handling already-closed pipes
        for pipe in self.parent_pipes:
            try:
                pipe.send(('close', None))
            except OSError:
                # Pipe is already closed, continue with cleanup
                pass
            try:
                pipe.close()
            except OSError:
                # Pipe is already closed, continue with cleanup
                pass

        # Wait for all processes to finish
        for process in self.processes:
            if process.is_alive():
                process.join()

        # Clean up shared memory
        self.shm.close()
        self.shm.unlink()

        # Mark as closed
        self._closed = True

    def render(self) -> Optional[list]:
        mode = self.render_mode
        if mode is None:
            return

        frames = []
        max_screen_width = 1920  # Max width for display, adjust for screen size
        max_screen_height = 1080  # Max height for display, adjust for screen size

        try:
            # Collect frames from each agent
            for pipe in self.parent_pipes:
                pipe.send(('render', None))
                frame = pipe.recv()
                if frame is not None:
                    frames.append(frame)

            if mode == "human":
                num_agents = len(frames)
                original_frame_h, original_frame_w = frames[0].shape[:2]

                # Determine the optimal grid layout within max screen constraints
                columns = min(num_agents, max_screen_width // original_frame_w)
                rows = math.ceil(num_agents / columns)

                # Calculate potential display size with original frame size
                total_width = columns * original_frame_w
                total_height = rows * original_frame_h

                # Resize frames if display size exceeds max screen constraints
                scale_w = max_screen_width / total_width if total_width > max_screen_width else 1.0
                scale_h = max_screen_height / total_height if total_height > max_screen_height else 1.0
                scale = min(scale_w, scale_h)

                # Calculate resized frame dimensions and display window size
                frame_w = int(original_frame_w * scale)
                frame_h = int(original_frame_h * scale)
                display_width = min(columns * frame_w, max_screen_width)
                display_height = min(rows * frame_h, max_screen_height)

                # Initialize or resize the pygame window if needed
                if not self.screen or self.screen.get_size() != (display_width, display_height):
                    pygame.init()
                    self.screen = pygame.display.set_mode((display_width, display_height))
                    pygame.display.set_caption("Multi-Agent Environment")

                # Render each frame in its grid position
                for idx, frame in enumerate(frames):
                    if scale < 1.0:
                        # Resize the frame if scaling is necessary
                        frame = cv2.resize(frame, (frame_w, frame_h))

                    col = idx % columns
                    row = idx // columns
                    x, y = col * frame_w, row * frame_h

                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    self.screen.blit(surface, (x, y))  # Position frame within the grid

                pygame.display.flip()
                time.sleep(0.01)  # Brief pause to simulate real-time rendering

            elif mode == "rgb_array":
                return frames  # Return a list of frames for each agent

        except AttributeError:
            return None