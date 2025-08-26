import argparse
import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import create_default_paths

# Function to parse log file for policy updates
def parse_log_updates(log_path):
    with open(log_path, 'r') as file:
        lines = file.readlines()

    # Regex to clean and find timestamps
    timestamp_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]'
    timestamps = []

    for line in lines:
        if "Updated weights for policy" in line:
            match = re.search(timestamp_pattern, line)
            if match:
                time_str = match.group(0).strip('[]')
                timestamp = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
                timestamps.append(timestamp)

    return timestamps

def main(args):
    # Load Safety-Gymnasium data
    safety_gym_df = pd.read_csv(args.safety_gym_csv)
    # safety_gym_updates = safety_gym_df['Time/Update']
    # safety_gym_cumulative = np.cumsum(np.ones(len(safety_gym_updates)))
    safety_gym_time = safety_gym_df['Time/Total']
    safety_gym_update = safety_gym_df['Time/Update']

    # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 100)
    safety_gym_update = np.insert(safety_gym_update, 0, 100)

    # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 30)
    safety_gym_update = np.insert(safety_gym_update, 0, 30)
    #
    # # Extract time and FPS, and prepend zeros
    safety_gym_time = np.insert(safety_gym_time, 0, 10)
    safety_gym_update = np.insert(safety_gym_update, 0, 10)
    #
    # Calculate updates per interval and their cumulative count
    updates_per_interval = np.diff(safety_gym_time, prepend=0) / safety_gym_update
    cumulative_updates = np.cumsum(updates_per_interval)

    # Prepare time in minutes for plotting
    time_minutes = safety_gym_time / 60  # Convert seconds to minutes

    # Convert update times to a cumulative sum of time intervals
    # safety_gym_time = np.cumsum(safety_gym_updates)

    # Calculate cumulative updates for HASARD
    hasard_timestamps = parse_log_updates(args.hasard_log)
    start_time = hasard_timestamps[0]
    hasard_minutes = [(ts - start_time).total_seconds() / 60 for ts in hasard_timestamps]
    hasard_cumulative = np.arange(1, len(hasard_minutes) + 1)

    # Plotting
    plt.figure(figsize=(5, 4))
    plt.xlim(-2, 120)
    plt.plot(hasard_minutes, hasard_cumulative, label='HASARD', color='blue')
    plt.plot(time_minutes, cumulative_updates, label='Safety-Gymnasium', color='green')
    plt.yscale('log')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cumulative Updates (log scale)')
    # plt.title('Cumulative Policy Updates Over Time')
    plt.grid(True, ls=":")
    plt.legend()

    # Save to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    os.makedirs(folder, exist_ok=True)
    full_path = f'{folder}/policy_updates.pdf'
    plt.savefig(full_path, dpi=300)
    print(f"Plot saved to: {full_path}")
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot policy updates comparison between HASARD and Safety-Gymnasium.")

    # Create default paths dynamically
    default_fps_dir = create_default_paths(__file__, 'fps')

    parser.add_argument("--safety_gym_csv", type=str, 
                        default=os.path.join(default_fps_dir, 'SafetyPointGoal.csv'),
                        help="Path to Safety-Gymnasium CSV file")
    parser.add_argument("--hasard_log", type=str, 
                        default=os.path.join(default_fps_dir, 'VolcanicVenture_Updates.out'),
                        help="Path to HASARD log file")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
