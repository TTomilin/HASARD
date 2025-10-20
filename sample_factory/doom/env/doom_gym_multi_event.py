import json  # For safe serialization of 'info'
import math
import time
from multiprocessing import Process, Event, shared_memory, Array, Value
from time import sleep
from typing import Optional, Dict, Tuple, Any, List

import cv2
import gymnasium as gym
import numpy as np
import pygame
import vizdoom as vzd
from vizdoom import ScreenResolution, Mode

from sample_factory.doom.env.doom_gym import VizdoomEnv
from sample_factory.doom.env.wrappers.reward_calculators import create_reward_calculator

def collect_agent_stats(game, reward, agent_id, episode_id, step_id, scenario_name=None):
    """
    Collect relevant stats from a VizDoom game instance for logging.
    Only collects basic stats and scenario-specific stats that are actually needed.

    Args:
        game: VizDoom game instance
        reward: Current step reward
        agent_id: Agent identifier
        episode_id: Episode identifier
        step_id: Step identifier within episode
        scenario_name: Name of the scenario for task-specific stats

    Returns:
        Dictionary containing agent stats
    """
    stats = {
        'agent_id': agent_id,
        'episode_id': episode_id,
        'step_id': step_id,
        'reward': reward,
    }

    try:
        # Basic stats for all tasks
        stats['health'] = game.get_game_variable(vzd.GameVariable.HEALTH)
        stats['armor'] = game.get_game_variable(vzd.GameVariable.ARMOR)

        # Add scenario-specific stats based on the task
        if scenario_name == 'remedy_rush':
            # Remedy Rush: cost tracking and goggles
            stats['cost'] = game.get_game_variable(vzd.GameVariable.USER1)
            stats['goggles_obtained'] = game.get_game_variable(vzd.GameVariable.USER2)
        elif scenario_name == 'volcanic_venture':
            # Volcanic Venture: health loss as cost
            pass  # Health is already tracked above
        elif scenario_name == 'armament_burden':
            # Armament Burden: weapon and delivery tracking
            stats['weapon_id'] = game.get_game_variable(vzd.GameVariable.USER1)
            stats['weapons_carried'] = game.get_game_variable(vzd.GameVariable.USER2)
            stats['in_delivery_zone'] = game.get_game_variable(vzd.GameVariable.USER3)
            stats['discarded'] = game.get_game_variable(vzd.GameVariable.USER4)
            stats['decoys_carried'] = game.get_game_variable(vzd.GameVariable.USER6)
        elif scenario_name == 'collateral_damage':
            # Collateral Damage: civilian casualty cost
            stats['civilian_cost'] = game.get_game_variable(vzd.GameVariable.USER1)
        elif scenario_name == 'detonators_dilemma':
            # Detonators Dilemma: unsafe detonation cost
            stats['unsafe_cost'] = game.get_game_variable(vzd.GameVariable.USER1)
        elif scenario_name == 'precipice_plunge':
            # Precipice Plunge: restart flag and position
            stats['restart_flag'] = game.get_game_variable(vzd.GameVariable.USER1)
            stats['position_z'] = game.get_game_variable(vzd.GameVariable.POSITION_Z)

    except Exception as e:
        # If any game variable is not available, set to 0
        print(f"Warning: Could not collect some stats for agent {agent_id}: {e}")
        for key in ['health', 'armor']:
            if key not in stats:
                stats[key] = 0

    return stats


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


def game_process(config_path, resolution, timeout, skip_frames, shared_command, step_event, all_done_event,
                 num_completed, num_agents, instance_id, is_host, port, shm_name, obs_shape, worker_idx, env_id,
                 netmode, async_mode, ticrate, reward_config):
    last_cycle = -1
    role = "HOST" if is_host else "PEER"
    print(f"[Worker {worker_idx}, Env {env_id}] Starting VizDoom {role} (Agent {instance_id}) on port {port}")

    # Initialize VizDoom game
    game = vzd.DoomGame()
    game.load_config(config_path)

    # Disable rendering and sound
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_console_enabled(False)
    game.set_screen_resolution(get_screen_resolution(resolution))
    game.set_episode_timeout(timeout)
    # Configure game mode based on async_mode parameter
    if async_mode:
        game.set_mode(Mode.ASYNC_PLAYER)  # Use ASYNC_PLAYER for multiplayer
    else:
        game.set_mode(Mode.PLAYER)
    game.set_ticrate(ticrate)

    if is_host:
        # Host game instance
        print(f"[Worker {worker_idx}, Env {env_id}] Configuring HOST for {num_agents} agents on port {port}")
        game.add_game_args(
            f"-host {num_agents} -port {port} -netmode {netmode} +timelimit 10.0 +sv_spawnfarthest 1"
        )
        game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
        game.add_game_args(f"+playernumber {instance_id}")
    else:
        # Join game instance
        print(f"[Worker {worker_idx}, Env {env_id}] Configuring PEER (Agent {instance_id}) to join port {port}")
        game.add_game_args(f"-join 127.0.0.1 -port {port} -netmode {netmode}")
        game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")
        game.add_game_args(f"+playernumber {instance_id}")

    print(f"[Worker {worker_idx}, Env {env_id}] Initializing VizDoom {role} (Agent {instance_id})...")
    game.init()
    print(f"[Worker {worker_idx}, Env {env_id}] VizDoom {role} (Agent {instance_id}) initialization complete!")

    # Connect to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    observations = np.ndarray(obs_shape, dtype=np.uint8, buffer=existing_shm.buf)

    reward_calculator = create_reward_calculator(reward_config)

    episode_id = 0
    step_id = 0
    role = "[HOST]" if is_host else "[PEER]"
    scenario = reward_config.get("scenario")

    try:
        while True:
            # Wait for the step event
            step_event.wait()
            # Do not clear the step_event here, the main process will clear it if necessary

            cmd_bytes = shared_command['cmd'].value
            cmd = cmd_bytes.decode().strip()
            data = list(shared_command['data'][:])

            # print(f"{role} Received command: {cmd} at step {step_id}")

            if cmd == 'step':
                action = data
                terminated = game.is_player_dead()
                if not terminated:
                    if skip_frames is not None:
                        reward = game.make_action(action, skip_frames)
                    else:
                        reward = game.make_action(action)
                    state = game.get_state()

                    # Calculate the reward
                    reward = reward_calculator.calculate_reward(game, reward)

                    if state and state.screen_buffer is not None:
                        observation = np.transpose(state.screen_buffer, (1, 2, 0))
                    else:
                        # print(f"{role} No state at process step {step_id}, env step {game.get_episode_time()}.")
                        observation = np.zeros(obs_shape[1:], dtype=np.uint8)

                    # Collect relevant stats from the game
                    stats = collect_agent_stats(game, reward, instance_id, episode_id, step_id, scenario)
                    info = {
                        "num_frames": skip_frames if skip_frames is not None else 1,
                        "agent_stats": stats
                    }
                else:
                    reward = 0.0
                    observation = np.zeros(obs_shape[1:], dtype=np.uint8)

                    # Collect basic stats even when terminated
                    stats = collect_agent_stats(game, reward, instance_id, episode_id, step_id, scenario)
                    info = {
                        "num_frames": skip_frames if skip_frames is not None else 1,
                        "agent_stats": stats
                    }

                # Write observation into shared memory
                observations[instance_id] = observation

                # Serialize 'info' as JSON string with safe truncation
                try:
                    info_json = json.dumps(info)
                    info_bytes = info_json.encode()

                    # If JSON is too large, try to create a minimal version
                    if len(info_bytes) > 1024:  # Increased buffer size
                        # Create minimal info with just essential data
                        minimal_info = {
                            "num_frames": info.get("num_frames", 1),
                            "agent_stats": {
                                "agent_id": info.get("agent_stats", {}).get("agent_id", instance_id),
                                "reward": info.get("agent_stats", {}).get("reward", reward),
                                "health": info.get("agent_stats", {}).get("health", 0),
                                "armor": info.get("agent_stats", {}).get("armor", 0)
                            }
                        }
                        info_json = json.dumps(minimal_info)
                        info_bytes = info_json.encode()

                    # Truncate if still too large, but ensure valid JSON
                    if len(info_bytes) > 1024:
                        info_bytes = info_bytes[:1020]  # Leave room for closing
                        # Try to find a safe truncation point (end of a complete field)
                        safe_end = info_bytes.rfind(b',')
                        if safe_end > 500:  # Only if we have substantial content
                            info_bytes = info_bytes[:safe_end] + b'}'
                        else:
                            # Fallback to minimal JSON
                            fallback_info = {"num_frames": 1, "error": "info_too_large"}
                            info_bytes = json.dumps(fallback_info).encode()

                    # Pad with null bytes to fill buffer
                    info_bytes += b'\x00' * (1024 - len(info_bytes))
                    shared_command['info'][:] = info_bytes

                except Exception as e:
                    # Fallback to minimal info if JSON serialization fails
                    fallback_info = {"num_frames": 1, "error": f"serialization_failed: {str(e)[:50]}"}
                    fallback_json = json.dumps(fallback_info)
                    fallback_bytes = fallback_json.encode()
                    fallback_bytes += b'\x00' * (1024 - len(fallback_bytes))
                    shared_command['info'][:] = fallback_bytes

                # Write results to shared_command
                shared_command['reward'].value = reward
                shared_command['terminated'].value = terminated

                # Increment the shared counter
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish steps
                        all_done_event.set()
                # Continue to next iteration, waiting for next step_event

                step_id += 1

            elif cmd == 'reset':
                # Reset the environment
                # print(f"BEFORE: {role} Episode finished: {game.is_episode_finished()}. Is new episode: {game.is_new_episode()}. Step ID: {step_id}, Episode ID: {episode_id}. Agent health: {game.get_game_variable(vzd.GameVariable.HEALTH)}")
                sleep(0.01)  # Give some time for the game to reset
                game.new_episode()
                game.respawn_player()
                # print(f"AFTER: {role} Episode finished: {game.is_episode_finished()}. Is new episode: {game.is_new_episode()}. Step ID: {step_id}, Episode ID: {episode_id}. Agent health: {game.get_game_variable(vzd.GameVariable.HEALTH)}")

                # Get initial state
                state = game.get_state()
                if state and state.screen_buffer is not None:
                    observation = np.transpose(state.screen_buffer, (1, 2, 0))
                else:
                    observation = np.zeros(obs_shape[1:], dtype=np.uint8)
                # Write observation into shared memory
                observations[instance_id] = observation

                # Reset reward calculator for new episode
                reward_calculator.reset(game)

                # Increment the shared counter
                with num_completed.get_lock():
                    num_completed.value += 1
                    if num_completed.value == num_agents:
                        # Last agent to finish reset
                        all_done_event.set()

                episode_id += 1
                step_id = 0

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
            scenario: str,
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
            async_mode=True,
            record_to=None,
            env_modification: str = None,
            resolution: str = "160x120",
            seed: Optional[int] = None,
            render_mode: Optional[str] = None,
            num_agents: int = 2,
            host_address: str = "127.0.0.1",
            port: int = 5029,
            env_config=None,
            netmode: int = 0,
            ticrate: int = 100,
            reward_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config_file, action_space, safety_bound, unsafe_reward, timeout, scenario, level, constraint,
                         coord_limits, max_histogram_length, show_automap, use_depth_buffer, render_depth_buffer,
                         render_with_bounding_boxes, segment_objects, skip_frames, async_mode, record_to,
                         env_modification, resolution, seed, render_mode)
        self.num_agents = num_agents
        self.host_address = host_address
        self.port = port
        self.resolution = resolution
        self.env_config = env_config
        self.netmode = netmode
        self.async_mode = async_mode
        self.ticrate = ticrate

        # Set default reward config if none provided - get scenario-specific configuration
        if reward_config is None:
            from sample_factory.doom.env.wrappers.reward_calculators import get_scenario_reward_config
            reward_config = get_scenario_reward_config(scenario, constraint)

        self.reward_config = reward_config

        # Extract worker and environment information for logging
        self.worker_idx = env_config.worker_index if env_config else -1
        self.env_id = env_config.env_id if env_config else -1

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
                'terminated': Value('b', False),
                'info': Array('c', 1024)  # Increased buffer size for JSON info
            }
            self.shared_commands.append(shared_command)

            # Create and start the game process, passing shared memory details
            process = Process(
                target=game_process,
                args=(
                    self.config_path, self.resolution, timeout * num_agents, self.skip_frames, shared_command,
                    self.step_event, self.all_done_event, self.num_completed, self.num_agents, i, is_host, self.port,
                    self.shm.name, multi_obs_shape, self.worker_idx, self.env_id, self.netmode, self.async_mode,
                    self.ticrate, self.reward_config
                ),
            )
            process.daemon = True
            process.start()
            self.processes.append(process)

        # Initialize step counting for timeout handling
        self._num_steps = 0
        self._timeout = timeout

        # Wait a bit to ensure all processes are ready
        time.sleep(1.0 + (self.port - 5029))  # Staggered delays

    def _barrier(self):
        with self.num_completed.get_lock():
            self.num_completed.value = 0
        self.step_event.set()
        self.all_done_event.wait()
        self.all_done_event.clear()
        self.step_event.clear()

    def reset(self, **kwargs):
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])

        self._num_steps = 0

        for i in range(self.num_agents):
            self.shared_commands[i]['cmd'].value = b'reset'
        self._barrier()

        observations = [self.observations[i].copy() for i in range(self.num_agents)]
        return observations, {}

    def step(self, actions) -> Tuple[Any, Any, Any, Any, Any]:
        # 1) Send actions to all children
        for i, action in enumerate(actions):
            sc = self.shared_commands[i]
            sc['cmd'].value = b'step'
            if len(action) != len(sc['data']):
                raise ValueError(
                    f"Action tuple length {len(action)} does not match expected {len(sc['data'])}."
                )
            sc['data'][:] = action

        # 2) Barrier (set -> wait -> clear)
        self._barrier()

        # 3) Gather results
        rewards, terminated, infos = [], [], []
        agent_stats_list = []

        for i in range(self.num_agents):
            sc = self.shared_commands[i]
            rewards.append(sc['reward'].value)
            # NOTE: child must write 'terminated' (engine end), not 'done'/'truncated'
            terminated.append(bool(sc['terminated'].value))

            info_bytes = sc['info'][:]
            info_str = bytes(info_bytes).decode().strip('\x00')

            # Safe JSON parsing with error handling
            if info_str:
                try:
                    info = json.loads(info_str)
                except json.JSONDecodeError as e:
                    # Handle corrupted JSON gracefully
                    print(f"Warning: JSON decode error for agent {i}: {e}")
                    print(f"Corrupted JSON string (first 100 chars): {info_str[:100]}")
                    # Provide fallback info
                    info = {
                        "num_frames": self.skip_frames if self.skip_frames is not None else 1,
                        "error": f"json_decode_failed: {str(e)[:50]}",
                        "agent_stats": {
                            "agent_id": i,
                            "reward": sc['reward'].value,
                            "health": 0,
                            "armor": 0
                        }
                    }
                except Exception as e:
                    # Handle any other decoding errors
                    print(f"Warning: Unexpected error decoding info for agent {i}: {e}")
                    info = {
                        "num_frames": self.skip_frames if self.skip_frames is not None else 1,
                        "error": f"decode_failed: {str(e)[:50]}",
                        "agent_stats": {
                            "agent_id": i,
                            "reward": sc['reward'].value,
                            "health": 0,
                            "armor": 0
                        }
                    }
            else:
                info = {}

            # Extract agent stats if available
            if 'agent_stats' in info:
                agent_stats_list.append(info['agent_stats'])

            infos.append(info)

        # Calculate combined stats from all agents
        combined_stats = self._calculate_combined_stats(agent_stats_list, rewards)

        # Structure stats for better organization in logging panels
        # Create separate agent stats and environment stats
        episode_extra_stats = {}

        # Add individual agent stats with proper agent IDs for separate panels
        for i, agent_stats in enumerate(agent_stats_list):
            if agent_stats:
                # Create agent-specific stats for separate panels (agent_0, agent_1, etc.)
                agent_key = f'agent_{i}'
                episode_extra_stats[f'individual_stats/{agent_key}/health'] = agent_stats.get('health', 0)
                episode_extra_stats[f'individual_stats/{agent_key}/armor'] = agent_stats.get('armor', 0)
                episode_extra_stats[f'individual_stats/{agent_key}/reward'] = rewards[i] if i < len(rewards) else 0

                # Add scenario-specific stats for each agent
                for stat_name, stat_value in agent_stats.items():
                    if stat_name not in ['agent_id', 'episode_id', 'step_id', 'reward', 'health', 'armor']:
                        episode_extra_stats[f'individual_stats/{agent_key}/{stat_name}'] = stat_value

        # Add environment-level stats (these are more appropriate as env_stats rather than policy_stats)
        episode_extra_stats['env_stats/num_agents'] = combined_stats.get('num_agents', 0)
        episode_extra_stats['env_stats/alive_agents'] = combined_stats.get('alive_agents', 0)
        episode_extra_stats['env_stats/dead_agents'] = combined_stats.get('dead_agents', 0)

        # Add team-level aggregated stats (these go under team_stats)
        episode_extra_stats['combined_stats/total_reward'] = combined_stats.get('total_reward', 0)
        episode_extra_stats['combined_stats/avg_reward'] = combined_stats.get('avg_reward', 0)
        episode_extra_stats['combined_stats/min_reward'] = combined_stats.get('min_reward', 0)
        episode_extra_stats['combined_stats/max_reward'] = combined_stats.get('max_reward', 0)
        episode_extra_stats['combined_stats/total_health'] = combined_stats.get('total_health', 0)
        episode_extra_stats['combined_stats/avg_health'] = combined_stats.get('avg_health', 0)
        episode_extra_stats['combined_stats/total_armor'] = combined_stats.get('total_armor', 0)
        episode_extra_stats['combined_stats/avg_armor'] = combined_stats.get('avg_armor', 0)

        # Add scenario-specific combined stats
        for key, value in combined_stats.items():
            if key.startswith('avg_') and key not in ['avg_reward', 'avg_health', 'avg_armor']:
                episode_extra_stats[f'combined_stats/{key}'] = value

        # Add the structured stats to each agent's info for logging
        for i, info in enumerate(infos):
            info['episode_extra_stats'] = episode_extra_stats.copy()
            info['episode_extra_stats']['agent_id'] = i

        observations = [self.observations[i].copy() for i in range(self.num_agents)]

        # 4) High-level truncation (timeout owned by this env)
        frames_advanced = infos[0].get("num_frames", self.skip_frames)
        self._num_steps += int(frames_advanced)

        timeout_reached = (self._timeout is not None) and (self._timeout > 0) and (self._num_steps >= self._timeout)
        truncated = [bool(timeout_reached) for _ in range(self.num_agents)]
        if timeout_reached:
            for info in infos:
                info["TimeLimit.truncated"] = True  # Gymnasium convention

        # 5) Return — do NOT reset here. Let the wrapper/caller trigger reset on (terminated | truncated).
        return observations, rewards, terminated, truncated, infos

    def close(self):
        # Set 'close' command for all agents
        for shared_command in self.shared_commands:
            shared_command['cmd'].value = b'close'
        # Signal all agents to proceed
        self.step_event.set()
        # Wait for processes to finish
        for process in self.processes:
            process.join()

        # Clean up pygame resources
        if hasattr(self, 'screen') and self.screen is not None:
            pygame.quit()
            self.screen = None

        # Clean up shared memory
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            # Shared memory already cleaned up, ignore
            pass

    def _calculate_combined_stats(self, agent_stats_list, rewards):
        """
        Calculate combined statistics across all agents.

        Args:
            agent_stats_list: List of individual agent stats dictionaries
            rewards: List of rewards for each agent

        Returns:
            Dictionary containing combined stats
        """
        if not agent_stats_list:
            return {}

        num_agents = len(agent_stats_list)
        combined = {
            'num_agents': num_agents,
            'total_reward': sum(rewards),
            'avg_reward': sum(rewards) / num_agents if num_agents > 0 else 0,
            'min_reward': min(rewards) if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
        }

        # Aggregate basic stats across all agents
        basic_stats = ['health', 'armor']

        for stat_name in basic_stats:
            values = []
            for agent_stats in agent_stats_list:
                if stat_name in agent_stats and agent_stats[stat_name] is not None:
                    values.append(agent_stats[stat_name])

            if values:
                combined[f'total_{stat_name}'] = sum(values)
                combined[f'avg_{stat_name}'] = sum(values) / len(values)
                combined[f'min_{stat_name}'] = min(values)
                combined[f'max_{stat_name}'] = max(values)
            else:
                combined[f'total_{stat_name}'] = 0
                combined[f'avg_{stat_name}'] = 0
                combined[f'min_{stat_name}'] = 0
                combined[f'max_{stat_name}'] = 0

        # Aggregate task-specific stats dynamically (episodic averages only)
        # Collect all unique stat names from agent stats, excluding basic stats
        all_stat_names = set()
        for agent_stats in agent_stats_list:
            all_stat_names.update(agent_stats.keys())

        # Remove basic stats and metadata that shouldn't be aggregated
        excluded_stats = {'agent_id', 'episode_id', 'step_id', 'reward', 'health', 'armor'}
        task_specific_stats = all_stat_names - excluded_stats

        for stat_name in task_specific_stats:
            values = []
            for agent_stats in agent_stats_list:
                if stat_name in agent_stats and agent_stats[stat_name] is not None:
                    values.append(agent_stats[stat_name])

            if values:
                # For task-specific stats, only log averages (not min/max)
                combined[f'avg_{stat_name}'] = sum(values) / len(values)

        # Count alive agents
        alive_agents = sum(1 for agent_stats in agent_stats_list if agent_stats.get('health', 0) > 0)
        combined['alive_agents'] = alive_agents
        combined['dead_agents'] = num_agents - alive_agents

        return combined

    def _calculate_grid_layout(self, num_agents: int, original_frame_w: int, max_screen_width: int = 1920) -> Tuple[
        int, int]:
        """Calculate optimal grid layout for multi-agent rendering."""
        columns = min(num_agents, max_screen_width // original_frame_w)
        rows = math.ceil(num_agents / columns)
        return columns, rows

    def _calculate_frame_scaling(self, columns: int, rows: int, original_frame_w: int, original_frame_h: int,
                                 max_screen_width: int = 1920, max_screen_height: int = 1080) -> Tuple[
        float, int, int, int, int]:
        """Calculate frame scaling and display dimensions."""
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

        return scale, frame_w, frame_h, display_width, display_height

    def _process_frame_for_position(self, frame: np.ndarray, idx: int, columns: int, scale: float,
                                    frame_w: int, frame_h: int) -> Tuple[np.ndarray, int, int]:
        """Process a frame for positioning in the grid."""
        # Resize the frame if scaling is necessary
        if scale < 1.0:
            frame = cv2.resize(frame, (frame_w, frame_h))

        # Calculate grid position
        col = idx % columns
        row = idx // columns
        x, y = col * frame_w, row * frame_h

        return frame, x, y

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
            frames = observations
            num_agents = len(frames)

            if num_agents == 0:
                return None if mode == "rgb_array" else None

            original_frame_h, original_frame_w = frames[0].shape[:2]

            # Calculate grid layout and frame scaling (common for both modes)
            columns, rows = self._calculate_grid_layout(num_agents, original_frame_w)
            scale, frame_w, frame_h, display_width, display_height = self._calculate_frame_scaling(
                columns, rows, original_frame_w, original_frame_h)

            if mode == "human":
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
                    processed_frame, x, y = self._process_frame_for_position(
                        frame, idx, columns, scale, frame_w, frame_h)

                    # Convert frame to surface
                    surface = pygame.surfarray.make_surface(processed_frame.swapaxes(0, 1))
                    self.screen.blit(surface, (x, y))  # Position frame within the grid

                pygame.display.flip()
                time.sleep(0.01)  # Brief pause to simulate real-time rendering

            elif mode == "rgb_array":
                num_channels = frames[0].shape[2] if len(frames[0].shape) > 2 else 1

                # Create the combined array
                if num_channels > 1:
                    combined_frame = np.zeros((display_height, display_width, num_channels), dtype=frames[0].dtype)
                else:
                    combined_frame = np.zeros((display_height, display_width), dtype=frames[0].dtype)

                # Place each frame in its grid position
                for idx, frame in enumerate(frames):
                    processed_frame, x, y = self._process_frame_for_position(
                        frame, idx, columns, scale, frame_w, frame_h)

                    # Ensure we don't exceed the combined frame boundaries
                    end_x = min(x + frame_w, display_width)
                    end_y = min(y + frame_h, display_height)
                    frame_end_x = end_x - x
                    frame_end_y = end_y - y

                    # Place the frame in the combined array
                    if num_channels > 1:
                        combined_frame[y:end_y, x:end_x] = processed_frame[:frame_end_y, :frame_end_x]
                    else:
                        combined_frame[y:end_y, x:end_x] = processed_frame[:frame_end_y, :frame_end_x]

                return combined_frame

        except Exception as e:
            print(f"Rendering Error: {e}")
            return None
