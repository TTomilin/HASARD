import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS, load_full_data, check_data_availability, create_default_paths, save_plot, create_common_parser
from sample_factory.doom.doom_utils import DOOM_ENVS


def main(args):
    # Check if any data is available for the specified path
    if not check_data_availability(args.input, args.algos[0], args.envs, args.seeds, args.metrics, args.level):
        print(f"Error: No data found at the specified path '{args.input}'. Please check that the path contains data for the specified environments, algorithms, seeds, metrics, and level.")
        return

    data = load_full_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level, args.hard_constraint)
    plot_metrics(data, args)


def reshape_data_if_needed(runs, num_seeds):
    """
    Helper function to reshape flat list data into separate runs per seed.

    Args:
        runs: Either a flat list of floats or a list of lists
        num_seeds: Number of seeds to split the data into

    Returns:
        List of lists, where each inner list represents one seed's data
    """
    # Check if runs is a flat list of floats (due to load_full_data using extend())
    # If so, we need to reshape it back into separate runs per seed
    if runs and isinstance(runs[0], (int, float)):
        # Flat list case - reshape into separate runs
        total_length = len(runs)

        # Only treat as single run if the data is clearly too small to be multi-seed
        # or if we have only one seed requested
        if num_seeds == 1 or total_length < num_seeds * 10:
            runs = [runs]
        else:
            # Try to split into approximately equal chunks
            # Calculate base chunk size and remainder
            base_chunk_size = total_length // num_seeds
            remainder = total_length % num_seeds

            # Split the data, distributing remainder across first few chunks
            new_runs = []
            start_idx = 0
            for i in range(num_seeds):
                # Add one extra element to first 'remainder' chunks
                chunk_size = base_chunk_size + (1 if i < remainder else 0)
                end_idx = start_idx + chunk_size
                new_runs.append(runs[start_idx:end_idx])
                start_idx = end_idx

            runs = new_runs

    # Now handle the case where runs might have different lengths
    if runs and hasattr(runs[0], '__len__'):
        min_length = min(len(run) for run in runs)
        for i in range(len(runs)):
            runs[i] = runs[i][:min_length]

    return runs


def plot_metrics(data, args):
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # Adjust figsize for better fit

    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.12, hspace=0.5, wspace=0.3)

    lines = []
    labels = []

    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}

    title_axes = [fig.add_subplot(3, 2, i + 1, frame_on=False) for i in range(6)]  # Update to match the number of environments
    for ax in title_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    for env_index, env in enumerate(args.envs):
        row = env_index // 2  # Keeps same row indexing
        col_base = (env_index % 2) * 2  # Same as before

        title_axes[env_index].set_title(TRANSLATIONS[env], fontsize=14)

        for metric_index, metric in enumerate(args.metrics):
            ax = axs[row, col_base + metric_index]
            for method in args.algos:
                key = (env, method, metric)
                if key in data and data[key]:
                    # all_runs = np.array(data[key])

                    # Hacky workaround for If some runs crashed and there is an uneven number of datapoints
                    runs = data[key]
                    runs = reshape_data_if_needed(runs, len(args.seeds))
                    all_runs = np.array(runs)

                    if all_runs.size == 0 or len(all_runs.shape) < 2:
                        continue

                    # The reward of PPOCost is logged with the cost subtracted from it
                    # We need to add it back for a fair comparison
                    if method == "PPOCost" and metric == "reward":
                        cost_key = (env, method, "cost")
                        if cost_key in data:
                            cost_runs = data[cost_key]
                            cost_runs = reshape_data_if_needed(cost_runs, len(args.seeds))
                            all_costs = np.array(cost_runs)
                            cost_scalar = doom_env_lookup[env].penalty_scaling
                            all_runs += all_costs * cost_scalar

                    num_data_points = all_runs.shape[1]
                    iterations_per_point = args.total_iterations / num_data_points
                    mean = np.mean(all_runs, axis=0)
                    ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                    x = np.arange(num_data_points) * iterations_per_point
                    line = ax.plot(x, mean, label=method)
                    ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)
                    ax.set_xlim(-args.total_iterations / 60, args.total_iterations)
                    ax.set_ylim(0, None)
                    ax.set_xlabel('Steps', fontsize=12)
                    ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)
                    if env_index == 1 and metric_index == 1:  # Adjust if needed
                        lines.append(line[0])
                        labels.append(method)
                    if metric == 'cost' and not args.hard_constraint:
                        threshold_line = ax.axhline(y=SAFETY_THRESHOLDS[env], color='red', linestyle='--',
                                                    label='Safety Threshold')
                        ax.text(0.5, SAFETY_THRESHOLDS[env], 'Safety Threshold', horizontalalignment='center',
                                verticalalignment='top', transform=ax.get_yaxis_transform(), fontsize=10,
                                style='italic', color='darkred')

    fontsize = 12 if len(args.algos) < 9 else 11
    fig.legend(lines, labels, loc='lower center', ncol=len(args.algos), fontsize=fontsize, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.0))

    # Save to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    file_name = 'hard' if args.hard_constraint else f'level_{args.level}'

    # Save using the common save_plot function
    save_plot(file_name, folder)


def common_plot_args() -> argparse.ArgumentParser:
    parser = create_common_parser("Plot metrics from structured data directory.")

    # Create default path dynamically
    default_main = create_default_paths(__file__, 'main')

    # Set defaults for common arguments
    parser.set_defaults(
        input=default_main,
        level=1,
        seeds=[1, 2, 3],
        envs=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
              "precipice_plunge", "detonators_dilemma"],
        metrics=['reward', 'cost'],
        hard_constraint=False
    )

    # Add specific arguments for this script
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O"],
                        help="Algorithms to download/plot")
    parser.add_argument("--total_iterations", type=int, default=int(5e8),
                        help="Total number of environment iterations corresponding to 500 data points")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
