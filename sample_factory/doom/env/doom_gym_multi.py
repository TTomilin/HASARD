import time
from multiprocessing import Process, Pipe
from typing import Optional, Dict, Tuple, Any

import gym
import gymnasium as gym
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution

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

        # Create pipes for communication with the game processes
        self.parent_pipes = []
        self.processes = []

        for i in range(self.num_agents):
            parent_conn, child_conn = Pipe()
            self.parent_pipes.append(parent_conn)

            # Determine if this is the host or joiner
            is_host = i == 0

            # Create and start the game process
            process = Process(
                target=self._game_process,
                args=(child_conn, i, is_host),
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

        # Wait a bit to ensure all processes are ready
        time.sleep(2.0)

    def _game_process(self, pipe, instance_id, is_host):
        # Each game instance runs in its own process
        game = vzd.DoomGame()
        game.load_config(self.config_path)

        # Disable rendering and sound
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

        # Set game mode to ASYNC_PLAYER for multiplayer
        game.set_mode(vzd.Mode.ASYNC_PLAYER)

        # Add multiplayer arguments
        if is_host:
            # Host game instance
            game.add_game_args(
                f"-host {self.num_agents} -port {self.port} -netmode 1 +timelimit 10.0 +sv_spawnfarthest 1")
            game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
            game.add_game_args(f"+playernumber {instance_id}")
        else:
            # Join game instance
            game.add_game_args(f"-join 127.0.0.1 -port {self.port} -netmode 1")
            game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
            game.add_game_args(f"+playernumber {instance_id}")

        game.init()

        # Main loop
        while True:
            cmd, data = pipe.recv()

            if cmd == 'step':
                action = data
                if not game.is_episode_finished():
                    game.make_action(action, self.skip_frames)
                    state = game.get_state()
                    reward = game.get_last_reward()
                    done = game.is_episode_finished()

                    if state and state.screen_buffer is not None:
                        observation = np.transpose(state.screen_buffer, (1, 2, 0))
                    else:
                        observation = np.zeros((240, 320, 3), dtype=np.uint8)

                    info = {}
                else:
                    reward = 0.0
                    done = True
                    observation = np.zeros((240, 320, 3), dtype=np.uint8)
                    info = {}

                pipe.send((observation, reward, done, info))

            elif cmd == 'reset':
                game.new_episode()
                state = game.get_state()
                if state and state.screen_buffer is not None:
                    observation = np.transpose(state.screen_buffer, (1, 2, 0))
                else:
                    observation = np.zeros((240, 320, 3), dtype=np.uint8)
                pipe.send(observation)

            elif cmd == 'close':
                game.close()
                pipe.close()
                break

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])

        for pipe in self.parent_pipes:
            pipe.send(('reset', None))

        observations = []
        for pipe in self.parent_pipes:
            obs = pipe.recv()
            observations.append(obs)

        # Return a list of observations for all agents
        return observations, {}

    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        # Send actions to each agent's process
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))

        observations = []
        rewards = []
        dones = []
        infos = []

        for pipe in self.parent_pipes:
            obs, reward, done, info = pipe.recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # Check if all agents are done
        all_done = all(dones)

        # Gym 0.26.0 changes
        terminated = all_done
        truncated = False

        # Return observations, rewards, dones, infos
        return observations, rewards, terminated, truncated, infos

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
            pipe.close()
        for process in self.processes:
            process.join()

    def render(self, mode='human'):
        pass  # Rendering can be implemented if needed
