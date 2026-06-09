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

from results.commons import save_plot

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

def cumulative_updates_from_df(df):
    """Compute cumulative policy updates from Time/Total and Time/Update columns."""
    t = df['Time/Total'].values
    u = df['Time/Update'].values
    t = np.insert(t, 0, [10, 30, 100])
    u = np.insert(u, 0, [10, 30, 100])
    updates_per_interval = np.diff(t, prepend=0) / u
    return t, np.cumsum(updates_per_interval)


def main(args):
    safety_gym_df = pd.read_csv(args.safety_gym_csv)
    crax_df = pd.read_csv(args.crax_csv)

    sg_time, sg_cumulative = cumulative_updates_from_df(safety_gym_df)
    crax_time, crax_cumulative = cumulative_updates_from_df(crax_df)

    hasard_timestamps = parse_log_updates(args.hasard_log)
    start_time = hasard_timestamps[0]
    hasard_minutes = [(ts - start_time).total_seconds() / 60 for ts in hasard_timestamps]
    hasard_cumulative = np.arange(1, len(hasard_minutes) + 1)

    plt.figure(figsize=(5, 4))
    plt.xlim(-2, 120)
    plt.plot(sg_time / 60, sg_cumulative, label='Safety-Gymnasium', color='green')
    plt.plot(hasard_minutes, hasard_cumulative, label='HASARD', color='blue')
    plt.plot(crax_time / 60, crax_cumulative, label='CRAX', color='orange')
    plt.yscale('log')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cumulative Updates (log scale)')
    plt.grid(True, ls=":")
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    save_plot('policy_updates', folder)


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot policy updates comparison between HASARD, CRAX, and Safety-Gymnasium.")

    from pathlib import Path
    default_fps_dir = Path(__file__).parent.parent.resolve() / 'data' / 'fps'

    parser.add_argument("--safety_gym_csv", type=str,
                        default=str(default_fps_dir / 'SafetyPointGoal.csv'),
                        help="Path to Safety-Gymnasium CSV file")
    parser.add_argument("--crax_csv", type=str,
                        default=str(default_fps_dir / 'CRAX.csv'),
                        help="Path to CRAX CSV file")
    parser.add_argument("--hasard_log", type=str,
                        default=str(default_fps_dir / 'VolcanicVenture_Updates.out'),
                        help="Path to HASARD log file")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
