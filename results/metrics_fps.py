import pandas as pd

# Load the CSV file
df = pd.read_csv('data/fps/SafetyPointGoal.csv')

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