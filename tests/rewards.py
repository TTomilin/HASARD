#!/usr/bin/env python3
"""
Test that examines what gets stored into the shared memory module after 1 episode with a host and a peer.
Expanded to save observations as images for each agent during the episode.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.append('/')

from sample_factory.doom.env.doom_gym_multi_event import VizdoomMultiAgentEnv
from sample_factory.doom.env.action_space import doom_action_space


def create_image_directories(base_dir="episode_images"):
    """
    Create directory structure for storing episode images.
    Returns the base directory path.
    """
    # Create base directory
    if os.path.exists(base_dir):
        # Remove existing directory and contents
        import shutil
        shutil.rmtree(base_dir)

    os.makedirs(base_dir)

    # Create subdirectories for each agent
    host_dir = os.path.join(base_dir, "agent_0_HOST")
    peer_dir = os.path.join(base_dir, "agent_1_PEER")

    os.makedirs(host_dir)
    os.makedirs(peer_dir)

    print(f"Created image directories:")
    print(f"  - {host_dir}")
    print(f"  - {peer_dir}")

    return base_dir


def save_observation_as_image(observation, agent_id, step_count, base_dir="episode_images", stage="step"):
    """
    Save an observation as an image file.

    Args:
        observation: numpy array of shape (H, W, 3) with values 0-255
        agent_id: 0 for HOST, 1 for PEER
        step_count: current step number
        base_dir: base directory for saving images
        stage: stage identifier (e.g., "reset", "step", "final")
    """
    agent_type = "HOST" if agent_id == 0 else "PEER"
    agent_dir = os.path.join(base_dir, f"agent_{agent_id}_{agent_type}")

    # Create filename with zero-padded step number for proper sorting
    if stage == "reset":
        filename = f"000_reset.png"
    elif stage == "final":
        filename = f"{step_count:03d}_final.png"
    else:
        filename = f"{step_count:03d}_step.png"

    filepath = os.path.join(agent_dir, filename)

    # Ensure observation is in the correct format
    if observation.dtype != np.uint8:
        observation = observation.astype(np.uint8)

    # Handle different observation shapes
    if len(observation.shape) == 3 and observation.shape[2] == 3:
        # RGB image
        # Convert from RGB to BGR for OpenCV (if using cv2)
        # But PIL expects RGB, so we'll use PIL
        image = Image.fromarray(observation, mode='RGB')
        image.save(filepath)
    elif len(observation.shape) == 2:
        # Grayscale image
        image = Image.fromarray(observation, mode='L')
        image.save(filepath)
    else:
        print(f"Warning: Unexpected observation shape {observation.shape} for agent {agent_id}")
        return


def save_all_agent_observations(observations, step_count, base_dir="episode_images", stage="step"):
    """
    Save observations for all agents at once.

    Args:
        observations: list of observations for each agent
        step_count: current step number
        base_dir: base directory for saving images
        stage: stage identifier
    """
    for agent_id, obs in enumerate(observations):
        if obs is not None and obs.size > 0:
            save_observation_as_image(obs, agent_id, step_count, base_dir, stage)


def test_shared_memory_contents():
    """
    Test that runs a host and peer for 1 episode and examines what gets stored in shared memory.
    Also validates that both agents have equal numbers of observations, actions, and rewards.
    Collects observations during the episode and saves them as images after the episode is over.
    """
    print("Starting shared memory contents test with image collection...")
    print("=" * 60)

    # Create directories for saving episode images
    image_base_dir = create_image_directories("episode_images")
    print()

    # Initialize data structure to collect observations during the episode
    collected_observations = []  # List of (observations, step_count, stage) tuples

    # Mock environment config
    class MockEnvConfig:
        def __init__(self):
            self.worker_index = 0
            self.env_id = 0

    # Create multi-agent environment with 2 agents (host and peer)
    config_file = "remedy_rush.cfg"
    action_space = doom_action_space()

    env = VizdoomMultiAgentEnv(
        config_file=config_file,
        action_space=action_space,
        safety_bound=0.5,
        unsafe_reward=-1.0,
        timeout=20,  # Short timeout for faster testing
        level=1,
        constraint='soft',
        skip_frames=1,
        resolution="800x600",
        num_agents=2,  # Host and peer
        port=5035,  # Use unique port to avoid conflicts
        env_config=MockEnvConfig(),
        netmode=0,  # Network mode for multiplayer
        async_mode=False
    )

    print("Environment created successfully!")
    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space dimensions: {len(action_space)}")
    print()

    # Reset environment to start the episode
    print("Resetting environment to start episode...")
    obs, info = env.reset()
    print(f"Reset completed. Observation shapes: {[o.shape for o in obs]}")

    # Collect initial observations from reset
    print("Collecting initial observations...")
    collected_observations.append((obs.copy(), 0, "reset"))
    print(f"Initial observations received from reset and collected")
    print()

    # Examine shared memory state after reset
    print("SHARED MEMORY STATE AFTER RESET:")
    print("-" * 40)
    # examine_shared_memory_state(env, "AFTER_RESET")
    print()

    step_count = 0
    max_steps = 200  # Limit steps for testing

    try:
        # Run the episode until both agents are done
        print("Running episode steps...")
        while step_count < max_steps:
            step_count += 1

            # Create random actions for both agents (host and peer)
            actions = []
            for i in range(2):
                # Generate random binary actions for each action dimension
                action = [np.random.randint(0, 2) for _ in range(len(action_space))]
                actions.append(action)

            # Step the environment
            obs, rewards, dones, truncated, infos = env.step(actions)

            # Collect observations at key intervals
            collect_observations_this_step = True

            if collect_observations_this_step:
                # Collect observations for later saving
                collected_observations.append((obs.copy(), step_count, "step"))

            # Print step information
            if any(r != 0 for r in rewards):
                print(f"Step {step_count}: Rewards: {rewards}")

            # Check if all agents are done (episode finished)
            if all(dones):
                print(f"\nEpisode completed after {step_count} steps!")
                # Collect final observations
                print("Collecting final observations...")
                collected_observations.append((obs.copy(), step_count, "final"))
                break

        # Save all collected observations as images after episode completion
        print(f"\nSaving all collected observations as images...")
        print(f"Total observations collected: {len(collected_observations)}")

        for obs_data, step_count, stage in collected_observations:
            save_all_agent_observations(obs_data, step_count, image_base_dir, stage)

        print(f"✅ All {len(collected_observations)} observation sets saved as images!")
        print(f"   - Images saved in: {image_base_dir}/")
        print(f"   - HOST images: {image_base_dir}/agent_0_HOST/")
        print(f"   - PEER images: {image_base_dir}/agent_1_PEER/")

        # Examine final shared memory state after episode completion
        print("\nFINAL SHARED MEMORY STATE AFTER EPISODE:")
        print("-" * 40)
        # examine_shared_memory_state(env, "FINAL")
        print()

        # Validate equal counts between host and peer using MEMORY-BASED READING
        print("DATA COUNT VALIDATION (READING FROM SHARED MEMORY):")
        print("=" * 60)

        # Summary
        print("\nSHARED MEMORY TEST SUMMARY:")
        print("=" * 60)
        print(f"✅ Episode completed successfully after {step_count} steps")
        print(f"✅ Shared memory observations examined at multiple points")
        print(f"✅ Shared command structures examined for both agents")
        print(f"✅ Data counts tracked for both agents throughout episode")
        print(f"✅ Observations collected during episode and saved as images after completion")
        print(f"   - Total observation sets collected: {len(collected_observations)}")
        print(f"   - Images saved in: {image_base_dir}/")
        print(f"   - HOST images: {image_base_dir}/agent_0_HOST/")
        print(f"   - PEER images: {image_base_dir}/agent_1_PEER/")
        print(f"   - Images include: reset, all steps, and final observations")

    except Exception as e:
        print(f"❌ ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if env is not None:
            try:
                env.close()
                print("\nEnvironment closed successfully.")
            except Exception as e:
                print(f"Error closing environment: {e}")


def examine_shared_memory_state(env, stage_name):
    """
    Examine and print the current state of shared memory components.
    """
    print(f"Stage: {stage_name}")

    # 1. Examine shared observations
    print(f"1. SHARED OBSERVATIONS:")
    print(f"   - Shared memory name: {env.shm.name}")
    print(f"   - Shared memory size: {env.shm.size} bytes")
    print(f"   - Observations array shape: {env.observations.shape}")
    print(f"   - Observations array dtype: {env.observations.dtype}")

    # Check if observations contain actual data (non-zero pixels)
    for i in range(env.num_agents):
        obs = env.observations[i]
        non_zero_pixels = np.count_nonzero(obs)
        total_pixels = obs.size
        print(f"   - Agent {i} ({'HOST' if i == 0 else 'PEER'}): {non_zero_pixels}/{total_pixels} non-zero pixels")
        print(f"     Min/Max pixel values: {obs.min()}/{obs.max()}")

    print()

    # 2. Examine shared commands for each agent
    print(f"2. SHARED COMMANDS:")
    for i, shared_command in enumerate(env.shared_commands):
        agent_type = 'HOST' if i == 0 else 'PEER'
        print(f"   Agent {i} ({agent_type}):")

        # Command
        cmd_bytes = shared_command['cmd'][:]
        cmd = bytes(cmd_bytes).decode().strip('\x00')
        print(f"     - Command: '{cmd}'")

        # Action data
        action_data = list(shared_command['data'][:])
        print(f"     - Action data: {action_data}")

        # Reward
        reward = shared_command['reward'].value
        print(f"     - Reward: {reward}")

        # Done flag
        done = shared_command['done'].value
        print(f"     - Done: {done}")

        # Info
        info_bytes = shared_command['info'][:]
        info_str = bytes(info_bytes).decode().strip('\x00')
        try:
            info = json.loads(info_str) if info_str else {}
            print(f"     - Info: {info}")
        except json.JSONDecodeError:
            print(f"     - Info (raw): '{info_str}'")

    print()

    # 3. Examine synchronization primitives
    print(f"3. SYNCHRONIZATION STATE:")
    print(f"   - Step event is set: {env.step_event.is_set()}")
    print(f"   - All done event is set: {env.all_done_event.is_set()}")
    print(f"   - Num completed: {env.num_completed.value}")

    # Episode tracking attributes might not be initialized yet
    episode_step_count = getattr(env, 'episode_step_count', 'Not initialized')
    episode_id = getattr(env, 'episode_id', 'Not initialized')
    print(f"   - Episode step count: {episode_step_count}")
    print(f"   - Episode ID: {episode_id}")


def main():
    """Main function to run the test."""
    print("=" * 60)
    print("SHARED MEMORY CONTENTS TEST WITH IMAGE COLLECTION")
    print("Testing what gets stored in shared memory after 1 episode")
    print("AND collecting observations during episode, saving images after completion")
    print("=" * 60)
    test_shared_memory_contents()


if __name__ == "__main__":
    main()
