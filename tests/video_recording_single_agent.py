#!/usr/bin/env python3

"""
Video recording test for multi-agent Doom environment.
This script demonstrates how to record gameplay videos using RGB array rendering.
"""

import sys
import os
import shutil
from pathlib import Path

from sample_factory.doom.env.doom_utils import make_doom_env_impl

# Add the parent directory to the path so we can import sample_factory
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

def test_rgb_array_rendering():
    """Test RGB array rendering (non-interactive) with video recording."""
    print("Multi-Agent Doom Video Recording Test")
    print("=" * 50)

    try:
        from sample_factory.doom.env.doom_utils import doom_env_by_name, make_doom_ma_env_impl
        from sample_factory.doom.train_vizdoom import parse_vizdoom_cfg
        from sample_factory.doom.env.wrappers.record_video import RecordVideo
        import numpy as np

        # Get environment spec
        spec = doom_env_by_name("remedy_rush")
        print(f"✓ Environment spec: {spec.name} with {spec.num_agents} agents")

        # Create test configuration
        cfg = parse_vizdoom_cfg(["--env=remedy_rush"])

        # Create mock env_config
        class MockEnvConfig:
            def __init__(self):
                self.env_id = 1  # Different port

        env_config = MockEnvConfig()

        # Create environment with rgb_array rendering
        print("Creating environment with RGB array rendering...")

        env = make_doom_env_impl(
            spec,
            cfg=cfg,
            env_config=env_config,
            render_mode="rgb_array"  # Changed to rgb_array for video recording
        )

        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of agents: {getattr(env, 'num_agents', 'Unknown')}")
        print(f"  Render mode: {env.render_mode}")

        video_dir = "test_videos"
        # Remove existing video directory and all its contents
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
            print(f"✓ Cleaned existing video directory: {video_dir}")

        # Wrap environment with video recording
        # Record every episode (lambda always returns True)
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: episode_id == 0,  # Record all episodes
            name_prefix="gameplay_recording",
            disable_logger=False,
        )

        episodes = 1
        ep_length = 2100
        print(f"\nRecording {episodes} episodes to {video_dir}/")

        # Record multiple episodes
        for i in range(episodes):
            print(f"Recording episode {i+1}/{episodes}...")
            env.reset()
            done = False
            step_count = 0
            episode_reward = 0.0

            while not done and step_count < ep_length:  # Limit steps to avoid very long videos
                action = env.action_space.sample()
                obs, reward, done, truncated, infos = env.step(action)

                # Track rewards for this episode
                episode_reward += reward

                done = done or truncated
                step_count += 1

            print(f"  Episode {i+1} completed: {step_count} steps")
            print(f"  Episode rewards: {f'{episode_reward:.2f}'}")

        print(f"\n✓ Video recording test completed successfully")
        print(f"✓ Videos saved to {os.path.abspath(video_dir)}/")

        # List created video files
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if video_files:
                print(f"✓ Created video files:")
                for video_file in video_files:
                    print(f"  - {video_file}")
            else:
                print("⚠ No video files found in output directory")

        # Clean up
        env.close()
        print("✓ Environment closed successfully")
        return True

    except Exception as e:
        print(f"✗ Video recording test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run video recording test."""
    print("Multi-Agent Doom Video Recording Test")
    print("=" * 60)

    print("RGB Array Rendering with Video Recording")
    print("This will record gameplay videos without opening a display window.")

    if test_rgb_array_rendering():
        print("\n✓ Video recording test passed")
        print("\nVideo Recording Features:")
        print("- RGB array mode for headless video recording")
        print("- Multiple episode recording")
        print("- Automatic video file generation")
        print("- Episode reward tracking during recording")
        return 0
    else:
        print("\n✗ Video recording test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
