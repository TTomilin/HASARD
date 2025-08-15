#!/usr/bin/env python3

"""
Episode rendering test for multi-agent Doom environment.
This script demonstrates how to render a multi-agent episode in real-time
and print rewards after each episode.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import sample_factory
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))


def test_real_time_rendering():
    """Test real-time rendering of a multi-agent episode."""
    print("Multi-Agent Doom Real-Time Rendering Test")
    print("=" * 50)

    try:
        from sample_factory.doom.env.doom_utils import doom_env_by_name, make_doom_ma_env_impl
        from sample_factory.doom.train_vizdoom import parse_vizdoom_cfg
        import numpy as np

        # Get environment spec
        spec = doom_env_by_name("remedy_rush")
        print(f"✓ Environment spec: {spec.name} with {spec.num_agents} agents")

        # Create test configuration with human rendering
        cfg = parse_vizdoom_cfg([
            "--env=remedy_rush",
            "--resolution=800x600",  # Set resolution for human rendering
        ])

        # Create mock env_config
        class MockEnvConfig:
            def __init__(self):
                self.env_id = 0

        env_config = MockEnvConfig()

        # Create environment with human rendering mode
        print("Creating environment with real-time rendering...")
        env = make_doom_ma_env_impl(
            spec,
            cfg=cfg,
            env_config=env_config,
            num_agents=2,
            render_mode="human"  # Enable real-time rendering
        )

        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of agents: {getattr(env, 'num_agents', 'Unknown')}")
        print(f"  Render mode: {env.render_mode}")

        # Reset environment
        print("\nResetting environment...")
        env.reset()
        print(f"✓ Environment reset successful")

        # Render initial state
        env.render()

        print("\nStarting real-time episode...")
        print("Note: A pygame window should open showing the multi-agent game.")
        print("The agents will take random actions. Close the window or press Ctrl+C to stop.")
        print("Episode will run for 200 steps or until done.\n")

        episode_length = 2100
        step_count = 0
        episode_rewards = [0.0] * env.num_agents

        try:
            while step_count < episode_length:
                # Generate random actions for each agent
                actions = []
                for i in range(env.num_agents):
                    action = env.action_space.sample()
                    actions.append(action)

                # Take environment step
                obs, rewards, dones, truncated, infos = env.step(actions)

                # Update episode rewards
                for i, reward in enumerate(rewards):
                    episode_rewards[i] += reward

                # Render the current state
                env.render()

                step_count += 1

                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"Step {step_count}/{episode_length}")
                    print(f"  Current rewards: {[f'{r:.2f}' for r in episode_rewards]}")

                # Check if episode is done
                if any(dones):
                    print(f"\nEpisode finished at step {step_count}")
                    break

        except KeyboardInterrupt:
            print(f"\nRendering interrupted by user at step {step_count}")

        print(f"\nEpisode Summary:")
        print(f"  Total steps: {step_count}")
        print(f"  Final rewards: {[f'{r:.2f}' for r in episode_rewards]}")
        print(f"  Average reward per agent: {np.mean(episode_rewards):.2f}")

        # Clean up
        try:
            env.close()
            print("✓ Environment closed successfully")
        except Exception as e:
            print(f"Warning: Error closing environment: {e}")

        return True

    except Exception as e:
        print(f"✗ Real-time rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run episode rendering test."""
    print("Multi-Agent Doom Episode Rendering Test")
    print("=" * 60)

    print("Real-Time Episode Rendering with Reward Printing")
    print("Opening pygame window showing the multi-agent game...")

    if test_real_time_rendering():
        print("✓ Episode rendering test passed")
        print("\nEpisode Rendering Features:")
        print("- Real-time pygame window with multi-agent view")
        print("- Reward tracking and printing during episodes")
        print("- Episode summary with final rewards")
        print("- Configurable resolution and frame rate")
        return 0
    else:
        print("✗ Episode rendering test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
