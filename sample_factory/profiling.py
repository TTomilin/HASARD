import pstats

def analyze_profiles(num_agents):
    for i in range(num_agents):
        profile_filename = f'game_process_{i}_profile.prof'
        print(f"\nProfiling results for game_process_{i}:")
        stats = pstats.Stats(profile_filename)
        stats.strip_dirs()
        stats.sort_stats('cumulative').print_stats(20)  # Top 20 functions

# Example usage:
# analyze_profiles(num_agents=2)

analyze_profiles(2)