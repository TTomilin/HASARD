import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import create_default_paths, save_plot


def main(args):
    # Load the CSV files
    safety_gym_df = pd.read_csv(args.safety_gym_csv)
    hasard_df = pd.read_csv(args.hasard_csv)

    # Extracting the relevant columns again
    safety_gym_time = safety_gym_df['Time/Total']
    safety_gym_fps = safety_gym_df['Time/FPS']

    safety_gym_time = safety_gym_df['Time/Total'].values
    safety_gym_fps = safety_gym_df['Time/FPS'].values

    # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 100)
    safety_gym_fps = np.insert(safety_gym_fps, 0, 100)

    # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 10)
    safety_gym_fps = np.insert(safety_gym_fps, 0, 100)

    # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 0)
    safety_gym_fps = np.insert(safety_gym_fps, 0, 0)

    # Calculate differences in time with correct initial padding
    time_diffs = np.diff(safety_gym_time, prepend=safety_gym_time[0])

    # Calculate cumulative frames correctly
    safety_gym_cumulative_frames = np.cumsum(safety_gym_fps * time_diffs)

    hasard_fps = hasard_df['PPO_volcanic_venture_level_1_soft_seed_3_20240810_012623_771642 - perf/_fps']

    # Recreate the plot with the updated data
    plt.figure(figsize=(5, 4))

    # Total runtime for the benchmark in seconds
    total_runtime_seconds = 2 * 3600 + 35 * 60 + 21

    # Assuming the steps are uniformly distributed over the total runtime
    hasard_time = np.linspace(0, total_runtime_seconds, len(hasard_fps))

    # Calculate cumulative frames by integrating the FPS over time
    # This calculation assumes that the FPS are provided at uniformly spaced time intervals.
    hasard_cumulative_frames = np.cumsum(hasard_fps * np.diff(hasard_time, prepend=0))

    # Convert seconds to minutes for plotting
    hasard_time_minutes = hasard_time / 60
    safety_gym_time_minutes = safety_gym_time / 60

    # Plotting Benchmark Cumulative Frames
    plt.plot(hasard_time_minutes, hasard_cumulative_frames, label='HASARD', color='blue')

    # Plotting Updated Safety-Gymnasium Cumulative Frames
    plt.plot(safety_gym_time_minutes, safety_gym_cumulative_frames, label='Safety-Gymnasium', color='green')

    plt.xlim(-2, 120)

    # Adding labels and title
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cumulative Frames (log scale)')
    # plt.title('Updated Cumulative Frames over Time for Benchmark and Safety-Gymnasium (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, ls=":")

    # Save and show the plot to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')

    # Save using the common save_plot function
    save_plot('FPS', folder)

    # Calculate average FPS
    average_fps_safety_gym = safety_gym_fps.mean()
    average_fps_hasard = hasard_fps.mean()

    # Print the average FPS values
    print(f"Average FPS for Safety-Gymnasium: {average_fps_safety_gym:.2f}")
    print(f"Average FPS for HASARD: {average_fps_hasard:.2f}")


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FPS comparison between HASARD and Safety-Gymnasium.")

    # Create default paths dynamically
    default_fps_dir = create_default_paths(__file__, 'fps')

    parser.add_argument("--safety_gym_csv", type=str, 
                        default=os.path.join(default_fps_dir, 'SafetyPointGoal.csv'),
                        help="Path to Safety-Gymnasium CSV file")
    parser.add_argument("--hasard_csv", type=str, 
                        default=os.path.join(default_fps_dir, 'VolcanicVenture.csv'),
                        help="Path to HASARD CSV file")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
