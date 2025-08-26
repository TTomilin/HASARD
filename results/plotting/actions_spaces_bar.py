import argparse
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import ENV_INITIALS, TRANSLATIONS, load_data, check_multiple_paths_data_availability, create_default_paths, save_plot, create_common_parser

TRANSLATIONS['data/main'] = 'Simplified Actions'


def process_metric(base_path: str, algo: str, env: str, seeds: List[int], metric: str, 
                   level: int, n_data_points: int, total_iterations: float) -> Dict[str, float]:
    """
    Process metric data for a specific environment and algorithm configuration.

    This function loads metric data for multiple seeds, extracts the last n_data_points
    from each seed's data (up to the total_iterations cutoff), and computes the mean
    and confidence interval across all collected data points.

    Args:
        base_path: Path to the data directory (e.g., 'data/main', 'data/full_actions')
        algo: Algorithm name (e.g., 'PPOLag')
        env: Environment name (e.g., 'armament_burden')
        seeds: List of random seeds to process
        metric: Metric name to extract (e.g., 'reward', 'cost')
        level: Environment difficulty level
        n_data_points: Number of final data points to average
        total_iterations: Maximum number of training iterations to consider

    Returns:
        Dictionary containing:
            - 'mean': Mean value of the metric across all seeds
            - 'ci': 95% confidence interval (1.96 * standard error)
    """
    recorded_total = 5e8  # Originally, data covers 500M steps
    metric_values = []

    for seed in seeds:
        # Load data for this specific seed and configuration
        data = load_data(base_path, algo, env, seed, level=level, metric_key=metric)
        if data is None or len(data) == 0:
            continue

        L = len(data)
        iters_per_point = recorded_total / L
        cutoff_index = int(total_iterations / iters_per_point)
        cutoff_index = min(cutoff_index, L)

        # Extract the final n_data_points from the data
        if cutoff_index >= n_data_points:
            window = data[cutoff_index - n_data_points:cutoff_index]
        else:
            window = data[:cutoff_index]

        if window:
            metric_values.extend(window)

    # Return zero values if no data was found
    if len(metric_values) == 0:
        return {'mean': 0.0, 'ci': 0.0}

    # Calculate mean and 95% confidence interval
    arr = np.array(metric_values)
    mean_val = np.mean(arr)
    ci = 1.96 * np.std(arr) / np.sqrt(len(arr))
    return {'mean': mean_val, 'ci': ci}


def process_data(action_spaces: List[str], algo: str, environments: List[str], seeds: List[int], 
                 metrics: List[str], level: int, n_data_points: int, total_iterations: float) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Process data for multiple environments, action spaces, and metrics.

    This function compares performance between simplified and full action spaces
    across different environments and metrics. It calculates the percentage difference
    between simplified and full action space performance.

    Args:
        action_spaces: List of action space paths (e.g., ['data/main', 'data/full_actions'])
        algo: Algorithm name (e.g., 'PPOLag')
        environments: List of environment names to process
        seeds: List of random seeds to process
        metrics: List of metrics to compare (e.g., ['reward', 'cost'])
        level: Environment difficulty level
        n_data_points: Number of final data points to average
        total_iterations: Maximum number of training iterations to consider

    Returns:
        Nested dictionary with structure:
        {
            env_name: {
                metric_name: {
                    'data/main': simplified_action_space_value,
                    'data/full_actions': full_action_space_value,
                    'diff': percentage_difference
                }
            }
        }
    """
    results = {}
    store = {}

    # Process all combinations of environments, action spaces, and metrics
    for env in environments:
        for base_path in action_spaces:
            for metric in metrics:
                out = process_metric(base_path, algo, env, seeds, metric, level, n_data_points, total_iterations)
                store[(env, base_path, metric)] = out['mean']

    # Calculate comparisons between simplified and full action spaces
    for env in environments:
        results[env] = {}
        for metric in metrics:
            # Use the actual paths from action_spaces instead of hardcoded strings
            simplified_path = action_spaces[0] if len(action_spaces) > 0 else None
            full_path = action_spaces[1] if len(action_spaces) > 1 else None

            simplified = store.get((env, simplified_path, metric), 0.0) if simplified_path else 0.0
            full = store.get((env, full_path, metric), 0.0) if full_path else 0.0

            # Calculate percentage difference: negative means simplified performs worse
            if simplified != 0.0:
                percent_diff = -((simplified - full) / simplified) * 100
            else:
                percent_diff = 0.0

            # Use normalized path names for the result keys to maintain compatibility
            simplified_key = os.path.basename(os.path.normpath(simplified_path)) if simplified_path else 'main'
            full_key = os.path.basename(os.path.normpath(full_path)) if full_path else 'full_actions'

            results[env][metric] = {
                simplified_key: simplified,
                full_key: full,
                'diff': percent_diff
            }
    return results


def plot_action_space_comparison(results: Dict[str, Dict[str, Dict[str, float]]], args: argparse.Namespace) -> None:
    """
    Create bar chart comparison between simplified and original action spaces.

    This function generates a side-by-side bar chart comparing reward and cost metrics
    between simplified and original action spaces across multiple environments.
    The plot includes safety threshold lines for the cost subplot.

    Args:
        results: Nested dictionary containing processed metric data with structure:
                {env_name: {metric_name: {'data/main': value, 'data/full_actions': value, 'diff': diff}}}
        args: Parsed command line arguments containing environment list, level, etc.

    Returns:
        None (saves plot to file and displays it)
    """
    envs = args.envs
    n_envs = len(envs)

    # Extract reward and cost values for both action spaces
    # Dynamically determine the keys from the results structure
    first_env = envs[0]
    available_keys = [k for k in results[first_env]['reward'].keys() if k != 'diff']

    # Assume first key is simplified, second is full (or use the same if only one)
    simplified_key = available_keys[0] if len(available_keys) > 0 else 'main'
    full_key = available_keys[1] if len(available_keys) > 1 else simplified_key

    reward_simpl = [results[env]['reward'][simplified_key] for env in envs]
    reward_full = [results[env]['reward'][full_key] for env in envs]
    cost_simpl = [results[env]['cost'][simplified_key] for env in envs]
    cost_full = [results[env]['cost'][full_key] for env in envs]

    # Set up bar chart positions and width
    x_positions = np.arange(n_envs)
    width = 0.4

    # Create subplots for reward and cost
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(7, 2), tight_layout=True)

    # Reward subplot - compare simplified vs original action spaces
    ax_r.bar(x_positions - width / 2, reward_simpl, width, label='Simplified Action Space')
    ax_r.bar(x_positions + width / 2, reward_full, width, label='Original Action Space')
    ax_r.set_ylabel("Reward")
    ax_r.set_xticks(x_positions)
    ax_r.set_xticklabels([ENV_INITIALS.get(env, env) for env in envs])

    # Cost subplot - compare simplified vs original action spaces
    ax_c.bar(x_positions - width / 2, cost_simpl, width, label='Simplified Action Space')
    ax_c.bar(x_positions + width / 2, cost_full, width, label='Original Action Space')
    ax_c.set_ylabel("Cost")
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels([ENV_INITIALS.get(env, env) for env in envs])

    # Add safety threshold lines (typical safety constraints in RL)
    ax_c.axhline(5, color='red', linestyle='--')
    ax_c.axhline(50, color='red', linestyle='--')

    # Create custom legend with safety threshold information
    custom_handles = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Simplified Action Space'),
        Line2D([0], [0], color='#ff7f0e', lw=4, label='Original Action Space'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Safety Threshold')
    ]

    # Position legend and adjust layout
    fig.legend(handles=custom_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    # Save plot to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    file_name = f'actions_level_{args.level}_bar'

    # Save using the common save_plot function
    save_plot(file_name, folder)


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate data processing and plotting.

    This function validates data availability, processes the metrics data for
    action space comparison, and generates the comparison plot.

    Args:
        args: Parsed command line arguments containing all configuration parameters
              including input paths, algorithm, environments, seeds, metrics, etc.

    Returns:
        None (generates and saves plot, may print error messages)
    """
    # Validate that data is available for the specified configuration
    if not check_multiple_paths_data_availability(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level):
        paths_str = "', '".join(args.inputs)
        print(f"Error: No data found at the specified paths ['{paths_str}']. "
              f"Please check that at least one path contains data for the specified "
              f"environments, algorithm, seeds, metrics, and level.")
        return

    # Process the data to compute metrics for action space comparison
    results = process_data(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level, args.n_data_points,
                           args.total_iterations)

    # Generate and save the comparison plot
    plot_action_space_comparison(results, args)


def get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for action space comparison plotting.

    This function sets up all command line arguments needed to configure the
    action space comparison analysis, including data paths, algorithm settings,
    environment selection, and plotting parameters.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all required
                                and optional arguments for the script
    """
    parser = create_common_parser(
        "Plot bar charts for reward & cost comparing simplified vs. original action spaces."
    )

    # Get the script's directory and construct default paths
    default_main, default_full_actions = create_default_paths(__file__, 'main', 'full_actions')

    # Set defaults for common arguments
    parser.set_defaults(
        inputs=[default_main, default_full_actions],
        level=1,
        method="PPOLag",
        seeds=[1, 2, 3],
        envs=["armament_burden", "volcanic_venture", "remedy_rush",
              "collateral_damage", "precipice_plunge", "detonators_dilemma"],
        metrics=["reward", "cost"],
        n_data_points=10
    )

    # Add specific arguments for this script
    parser.add_argument("--total_iterations", type=float, default=2e8,
                        help="Data cutoff in environment steps (e.g., 200M steps).")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
