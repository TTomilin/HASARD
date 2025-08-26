import pandas as pd
import os
import sys
import argparse

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import create_default_paths

def main(csv_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: No data found at the specified path '{csv_path}'. Please check that the path contains the required CSV file.")
        exit(1)

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract the 'Time/Update' column which should provide the time per update in seconds
    time_updates = df['Time/Update']

    # Calculate the total duration by summing up the intervals between updates
    total_duration = sum(time_updates)  # Summing all intervals if they represent individual update durations

    # Count the total number of updates
    total_updates = len(time_updates)

    # Calculate the frequency of updates per second
    update_frequency = total_updates / total_duration

    print(f"Total Duration: {total_duration} seconds")
    print(f"Total Updates: {total_updates}")
    print(f"Update Frequency: {update_frequency} updates per second")


def common_plot_args():
    parser = argparse.ArgumentParser(description="Analyze FPS data from CSV file.")

    # Create default path dynamically
    default_fps_dir = create_default_paths(__file__, 'fps')
    default_csv_path = os.path.join(default_fps_dir, 'SafetyPointGoal.csv')

    parser.add_argument("--csv_path", type=str, default=default_csv_path,
                        help="Path to the CSV file containing FPS data")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args.csv_path)
