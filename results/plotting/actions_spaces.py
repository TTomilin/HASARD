import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS, load_full_data, \
    check_multiple_paths_data_availability, create_default_paths, save_plot


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


def load_action_space_data(action_spaces: List[str], algo: str, environments: List[str],
                           seeds: List[int], metrics: List[str], level: int,
                           hard_constraint: bool) -> Dict[str, Dict[Tuple[str, str, str], List[float]]]:
    """
    Load training curve data for different action spaces.

    Args:
        action_spaces: List of action space paths (e.g., ['data/main', 'data/full_actions'])
        algo: Algorithm name (e.g., 'PPOLag')
        environments: List of environment names
        seeds: List of random seeds
        metrics: List of metrics to load (e.g., ['reward', 'cost'])
        level: Environment difficulty level
        hard_constraint: Whether to use hard constraint metrics

    Returns:
        Dictionary mapping action space paths to data dictionaries
    """
    action_space_data = {}

    for action_space in action_spaces:
        # Check if data is available for this action space
        if check_multiple_paths_data_availability([action_space], algo, environments, seeds, metrics, level):
            data = load_full_data(action_space, environments, [algo], seeds, metrics, level, hard_constraint)
            action_space_data[action_space] = data
        else:
            print(f"Warning: No data found for action space '{action_space}'. Skipping.")

    return action_space_data


def plot_action_space_training_curves(action_space_data: Dict[str, Dict[Tuple[str, str, str], List[float]]],
                                      args: argparse.Namespace) -> None:
    """
    Plot training curves comparing different action spaces across environments.

    Args:
        action_space_data: Dictionary mapping action space paths to training data
        args: Parsed command line arguments
    """
    if not action_space_data:
        print("Error: No data available for any action space.")
        return

    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # Adjust figsize for better fit

    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.12, hspace=0.5, wspace=0.3)

    lines = []
    labels = []

    title_axes = [fig.add_subplot(3, 2, i + 1, frame_on=False) for i in
                  range(6)]  # Update to match the number of environments
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

            # Plot each action space
            for action_space, data in action_space_data.items():
                key = (env, args.algo, metric)
                if key in data and data[key]:
                    runs = data[key]
                    runs = reshape_data_if_needed(runs, len(args.seeds))
                    all_runs = np.array(runs)

                    if all_runs.size == 0 or len(all_runs.shape) < 2:
                        continue

                    num_data_points = all_runs.shape[1]
                    iterations_per_point = args.total_iterations / num_data_points
                    mean = np.mean(all_runs, axis=0)
                    ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                    x = np.arange(num_data_points) * iterations_per_point

                    # Get readable label for action space
                    label = action_space.split('/')[-1]
                    label = "Original" if label == 'main' else "Simplfied"
                    label += " Action Space"

                    line = ax.plot(x, mean, label=label)
                    ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)

                    if env_index == 1 and metric_index == 1:  # Adjust if needed
                        lines.append(line[0])
                        labels.append(label)

            # Set axis properties once per subplot (outside the action space loop)
            ax.set_xlim(-args.total_iterations / 60, args.total_iterations)
            ax.set_ylim(0, None)
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)

            # Add safety threshold for cost metrics
            if metric == 'cost' and not args.hard_constraint:
                threshold_line = ax.axhline(y=SAFETY_THRESHOLDS[env], color='red', linestyle='--',
                                            label='Safety Threshold')
                ax.text(0.5, SAFETY_THRESHOLDS[env], 'Safety Threshold', horizontalalignment='center',
                        verticalalignment='top', transform=ax.get_yaxis_transform(), fontsize=10,
                        style='italic', color='darkred')

    fontsize = 12 if len(action_space_data) < 9 else 11
    fig.legend(lines, labels, loc='lower center', ncol=len(action_space_data), fontsize=fontsize, fancybox=True,
               shadow=True,
               bbox_to_anchor=(0.5, 0.0))

    # Save to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    file_name = 'hard' if args.hard_constraint else f'actions_level_{args.level}'

    # Save using the common save_plot function
    save_plot(file_name, folder)


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate data loading and plotting.

    Args:
        args: Parsed command line arguments containing all configuration
    """
    # Load data for different action spaces
    action_space_data = load_action_space_data(
        args.action_spaces, args.algo, args.envs, args.seeds,
        args.metrics, args.level, args.hard_constraint
    )

    if not action_space_data:
        print("Error: No data found for any action space. Please check your data paths and parameters.")
        return

    # Plot the training curves
    plot_action_space_training_curves(action_space_data, args)


def get_parser() -> argparse.Namespace:
    """
    Create and configure argument parser for the script.

    Returns:
        Configured ArgumentParser instance with all required arguments
    """
    parser = argparse.ArgumentParser(
        description="Plot training curves comparing different action spaces across environments."
    )

    # Create default paths
    default_main = create_default_paths(__file__, 'main')
    default_full_actions = create_default_paths(__file__, 'full_actions')

    parser.add_argument(
        "--action_spaces",
        type=str,
        nargs='+',
        default=[default_main, default_full_actions],
        help="Paths to different action space data directories"
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="PPOLag",
        help="Algorithm to plot (default: PPOLag)"
    )

    parser.add_argument(
        "--envs",
        type=str,
        nargs='+',
        default=["armament_burden", "volcanic_venture", "remedy_rush",
                 "collateral_damage", "precipice_plunge", "detonators_dilemma"],
        help="Environments to plot (default: all 6 environments)"
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help="Seeds to aggregate over (default: [1, 2, 3])"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=['reward', 'cost'],
        help="Metrics to plot (default: ['reward', 'cost'])"
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Environment difficulty level (default: 1)"
    )

    parser.add_argument(
        "--hard_constraint",
        action='store_true',
        help="Use hard constraint metrics"
    )

    parser.add_argument(
        "--total_iterations",
        type=float,
        default=5e8,
        help="Total number of environment iterations (default: 5e8)"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
