import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import re

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

# Load Safety-Gymnasium data
safety_gym_df = pd.read_csv('data/fps/SafetyPointGoal.csv')
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
log_path = '../data/fps/VolcanicVenture_Updates.out'
hasard_timestamps = parse_log_updates(log_path)
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
plt.savefig(f'figures/policy_updates.pdf', dpi=300)
plt.show()
