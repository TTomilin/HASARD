import math
import time
from multiprocessing import Process, Event, shared_memory, Array, Value
from ctypes import c_int
from typing import Optional, Dict, Tuple, Any, List

import cv2
import gym
import numpy as np
import pygame
import vizdoom as vzd
from vizdoom import ScreenResolution, Mode

from sample_factory.doom.env.doom_gym import VizdoomEnv

import json  # For safe serialization of 'info'

# Define screen resolutions
resolutions = {
    '1920x1080': ScreenResolution.RES_1920X1080,
    '1600x1200': ScreenResolution.RES_1600X1200,
    '1280x720': ScreenResolution.RES_1280X720,
    '800x600': ScreenResolution.RES_800X600,
    '640x480': ScreenResolution.RES_640X480,
    '320x240': ScreenResolution.RES_320X240,
    '160x120': ScreenResolution.RES_160X120
}

def get_screen_resolution(resolution: str) -> ScreenResolution:
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}')
    return resolutions[resolution]

def game_process(config_path, resolution, skip_frames, shared_command, step_event, all_done_event,
                 num_completed, num_agents, instance_id, is_host, port, shm_name, obs_shape):
    # Initialize VizDoom game
    game = vzd.DoomGame()
    game.load_config(config_path)

    # Disable rendering and sound
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_mode(Mode.ASYNC_PLAYER)  # Use ASYNC_PLAYER for multiplayer
    game.set_ticrate(1000)

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
        while True:
            # Wait for the step event
            step_event.wait()
            # Do not clear the step_event here, the main process will clear it if necessary

            cmd_bytes = shared_command['cmd'].value
            cmd = cmd_bytes.decode().strip()
            data = list(shared_command['data'][:])

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

                # Serialize 'info' as JSON string
                info_json = json.dumps(info)
                info_bytes = info_json.encode()[:256]  # Truncate if necessary
                info_bytes += b'\x00' * (256 - len(info_bytes))  # Pad with null bytes
                shared_command['info'][:] = info_bytes

                # Write results to shared_command
                shared_command['reward'].value = reward
                shared_command['done'].value = done

                # Increment the shared counter
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish steps
                        all_done_event.set()
                # Continue to next iteration, waiting for next step_event

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

                # Increment the shared counter
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish reset
                        all_done_event.set()
                # Continue to next iteration, waiting for next step_event

            elif cmd == 'close':
                game.close()
                existing_shm.close()
                break

            else:
                print(f"Unknown command: {cmd}")
                # Wait for the next command
    finally:
        pass  # Cleanup if necessary

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

        parts = resolution.lower().split("x")
        width = int(parts[0])
        height = int(parts[1])

        # Define observation and action spaces
        obs_shape = (height, width, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        action_dims = len(action_space)

        # Create shared memory for observations
        multi_obs_shape = (self.num_agents, height, width, 3)
        obs_size = np.prod(multi_obs_shape) * np.dtype(np.uint8).itemsize
        self.shm = shared_memory.SharedMemory(create=True, size=obs_size)
        self.observations = np.ndarray(multi_obs_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Create shared commands and events for synchronization
        self.shared_commands = []
        self.processes = []

        # Shared synchronization primitives
        self.step_event = Event()
        self.all_done_event = Event()
        self.num_completed = Value('i', 0)

        for i in range(self.num_agents):
            is_host = i == 0

            # Shared command dictionary using multiprocessing.Array and Value
            shared_command = {
                'cmd': Array('c', 10),  # Command string, max length 10
                'data': Array('i', action_dims),  # Array to hold each discrete action
                'reward': Value('d', 0.0),
                'done': Value('b', False),
                'info': Array('c', 256)  # Placeholder for info, adjust size as needed
            }
            self.shared_commands.append(shared_command)

            # Create and start the game process, passing shared memory details
            process = Process(
                target=game_process,
                args=(
                    self.config_path, self.resolution, self.skip_frames,
                    shared_command, self.step_event, self.all_done_event,
                    self.num_completed, self.num_agents, i, is_host, self.port,
                    self.shm.name, multi_obs_shape
                ),
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

        # Wait a bit to ensure all processes are ready
        time.sleep(1.0)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])

        # Set 'reset' command for all agents
        for shared_command in self.shared_commands:
            shared_command['cmd'].value = b'reset'
            shared_command['data'][:] = [0] * len(shared_command['data'])  # Clear data

        # Reset the shared counter
        with self.num_completed.get_lock():
            self.num_completed.value = 0

        # Signal all agents to proceed
        self.step_event.set()

        # Wait for all agents to complete reset
        self.all_done_event.wait()
        self.all_done_event.clear()

        # Clear the step_event so agents wait for the next command
        self.step_event.clear()

        observations = [self.observations[i].copy() for i in range(self.num_agents)]

        return observations, {}

    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        # Set commands and actions for all agents
        for i, action in enumerate(actions):
            shared_command = self.shared_commands[i]
            shared_command['cmd'].value = b'step'
            # Ensure action is a tuple or list of integers
            if len(action) != len(shared_command['data']):
                raise ValueError(f"Action tuple length {len(action)} does not match expected {len(shared_command['data'])}.")
            shared_command['data'][:] = action

        # Reset the shared counter
        with self.num_completed.get_lock():
            self.num_completed.value = 0

        # Signal all agents to proceed
        self.step_event.set()

        # Wait for all agents to complete
        self.all_done_event.wait()
        self.all_done_event.clear()

        # Clear the step_event so agents wait for the next command
        self.step_event.clear()

        # Collect results from shared memory
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_agents):
            shared_command = self.shared_commands[i]
            reward = shared_command['reward'].value
            done = shared_command['done'].value
            # Deserialize 'info' if necessary
            info_bytes = shared_command['info'][:]
            info_str = bytes(info_bytes).decode().strip('\x00')
            info = json.loads(info_str) if info_str else {}
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        observations = [self.observations[i].copy() for i in range(self.num_agents)]
        truncated = [False] * len(dones)  # Implement proper truncation if needed

        return observations, rewards, dones, truncated, infos

    def close(self):
        # Set 'close' command for all agents
        for shared_command in self.shared_commands:
            shared_command['cmd'].value = b'close'
        # Signal all agents to proceed
        self.step_event.set()
        # Wait for processes to finish
        for process in self.processes:
            process.join()

        # Clean up shared memory
        self.shm.close()
        self.shm.unlink()

    def render(self) -> Optional[List[np.ndarray]]:
        """
        Render the frames from all agents.

        Returns:
            Optional[List[np.ndarray]]: A list of frames for each agent if render_mode is 'rgb_array',
                                        otherwise renders the frames using pygame and returns None.
        """
        mode = self.render_mode
        if mode is None:
            return

        try:
            # Read observations from shared memory
            observations = [self.observations[i].copy() for i in range(self.num_agents)]

            if mode == "human":
                frames = observations
                num_agents = len(frames)
                if num_agents == 0:
                    return

                original_frame_h, original_frame_w = frames[0].shape[:2]

                # Determine the optimal grid layout within max screen constraints
                max_screen_width = 1920
                max_screen_height = 1080
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

                # Handle Pygame events to allow window closure
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        pygame.quit()
                        return

                # Render each frame in its grid position
                for idx, frame in enumerate(frames):
                    if scale < 1.0:
                        # Resize the frame if scaling is necessary
                        frame = cv2.resize(frame, (frame_w, frame_h))

                    col = idx % columns
                    row = idx // columns
                    x, y = col * frame_w, row * frame_h

                    # Convert frame to surface
                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    self.screen.blit(surface, (x, y))  # Position frame within the grid

                pygame.display.flip()
                time.sleep(0.01)  # Brief pause to simulate real-time rendering

            elif mode == "rgb_array":
                return observations  # Return a list of frames for each agent

        except Exception as e:
            print(f"Rendering Error: {e}")
            return None
