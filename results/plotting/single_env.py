import argparse
import json
import os
import sys
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS, create_default_paths, create_common_parser
from sample_factory.doom.doom_utils import DOOM_ENVS


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate data loading and metric plotting for a single environment.

    This function validates input arguments, loads experimental data, and generates
    plots showing metric evolution over training steps for the specified environment.

    Args:
        args: Parsed command line arguments containing:
            - inputs: List of data directory paths
            - total_iterations: List of total iterations for each input (must match inputs length)
            - env: Environment name
            - algo: Algorithm name
            - seeds: List of random seeds
            - metrics: List of metrics to plot
            - level: Environment difficulty level

    Returns:
        None (generates and saves plots)

    Raises:
        ValueError: If the number of total_iterations doesn't match the number of inputs
    """
    # Validate that total_iterations matches inputs
    if len(args.total_iterations) != len(args.inputs):
        raise ValueError("The number of total_iterations must match the number of inputs.")

    # Load experimental data
    data = load_data(args.inputs, args.envs[0], args.method, args.seeds, args.metrics, args.level)

    # Generate and save plots
    plot_metrics(data, args)


def load_data(base_paths: List[str], environment: str, method: str, seeds: List[int], 
              metrics: List[str], level: int) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Load experimental data from structured directories for multiple configurations.

    This function loads metric data (e.g., reward, cost) from multiple base paths
    and seeds, organizing the data by metric and base path. Each metric contains
    runs from different seeds for comparison and statistical analysis.

    Args:
        base_paths: List of base directory paths containing experimental data
        environment: Environment name (e.g., 'armament_burden')
        method: Algorithm/method name (e.g., 'PPOLag')
        seeds: List of random seeds to load data from
        metrics: List of metric names to load (e.g., ['reward', 'cost'])
        level: Environment difficulty level

    Returns:
        Nested dictionary with structure:
        {
            metric_name: {
                base_path: [
                    [run_data_seed1],
                    [run_data_seed2],
                    ...
                ]
            }
        }

    File Structure Expected:
        base_path/environment/method/level_X/seed_Y/metric.json

    Examples:
        >>> data = load_data(['data/main'], 'armament_burden', 'PPOLag', [1, 2], ['reward'], 1)
        >>> data['reward']['data/main']
        [[0.1, 0.2, 0.3, ...], [0.15, 0.25, 0.35, ...]]  # Two runs from seeds 1 and 2
    """
    data = {}

    # Initialize data structure for each metric
    for metric in metrics:
        data[metric] = {}

        # Load data from each base path
        for base_path in base_paths:
            runs = []

            # Load data from each seed
            for seed in seeds:
                file_path = os.path.join(base_path, environment, method, f"level_{level}", 
                                       f"seed_{seed}", f"{metric}.json")
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as file:
                            runs.append(json.load(file))
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error loading file {file_path}: {e}")
                else:
                    print(f"File not found: {file_path}")

            # Store runs for this base path
            data[metric][base_path] = runs

    return data


def plot_metrics(data: Dict[str, Dict[str, List[List[float]]]], args: argparse.Namespace) -> None:
    """
    Generate and save metric plots showing training progress over time.

    This function creates line plots with confidence intervals for each metric,
    comparing different experimental configurations. It handles special cases
    like PPOCost reward adjustment and safety threshold visualization.

    Args:
        data: Nested dictionary containing metric data with structure:
              {metric_name: {base_path: [[run_data_seed1], [run_data_seed2], ...]}}
        args: Parsed command line arguments containing:
            - metrics: List of metrics to plot
            - inputs: List of input directory paths
            - total_iterations: List of total iterations for each input
            - algo: Algorithm name (affects PPOCost reward calculation)
            - env: Environment name (for safety thresholds)
            - level: Environment difficulty level
            - hard_constraint: Whether using hard constraints

    Returns:
        None (generates and saves plot to figures/ directory)

    Plot Features:
        - Line plots with 95% confidence intervals
        - Safety threshold lines for cost metrics
        - PPOCost reward adjustment (adds scaled cost to reward)
        - Automatic axis scaling and labeling
        - Legend with translated labels
    """
    # Set plotting style
    plt.style.use('seaborn-v0_8-paper')
    num_metrics = len(args.metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 2.5))

    # Ensure axs is always a list for consistent indexing
    if num_metrics == 1:
        axs = [axs]

    # Create lookup for DOOM environment specifications
    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}

    total_iterations_dict = dict(zip(args.inputs, args.total_iterations))

    for metric_index, metric in enumerate(args.metrics):
        ax = axs[metric_index]
        lines = []
        labels = []

        # Compute global min_length across all runs for this metric
        all_lengths = []
        for base_path in args.inputs:
            runs = data[metric].get(base_path, [])
            for run in runs:
                all_lengths.append(len(run))
        if not all_lengths:
            continue  # No data for this metric
        global_min_length = min(all_lengths)

        for base_path in args.inputs:
            runs = data[metric].get(base_path, [])
            if runs:
                # Trim all runs to the global_min_length
                runs = [run[:global_min_length] for run in runs]
                all_runs = np.array(runs)
                if all_runs.size == 0 or len(all_runs.shape) < 2:
                    continue
                # Adjust for PPOCost reward if needed
                if args.method == "PPOCost" and metric == "reward":
                    cost_runs = data.get('cost', {}).get(base_path, [])
                    if cost_runs:
                        # Also trim cost_runs to global_min_length
                        cost_runs = [run[:global_min_length] for run in cost_runs]
                        all_costs = np.array(cost_runs)
                        cost_scalar = doom_env_lookup[args.envs[0]].penalty_scaling
                        all_runs += all_costs * cost_scalar
                mean = np.mean(all_runs, axis=0)
                ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                num_data_points = all_runs.shape[1]
                total_iterations = float(total_iterations_dict[base_path])
                iterations_per_point = total_iterations / num_data_points
                x = np.arange(num_data_points) * iterations_per_point
                # If there are multiple inputs, distinguish them; otherwise just show method name
                if len(args.inputs) > 1:
                    # Use method name + data source description
                    data_source = TRANSLATIONS.get(base_path, os.path.basename(base_path))
                    label = f"{args.method} ({data_source})"
                else:
                    # Single input: just show method name
                    label = args.method
                line = ax.plot(x, mean, label=label)
                ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)
                ax.set_xlim(-total_iterations / 120, total_iterations)
                ax.set_ylim(0, None)
                lines.append(line[0])
                labels.append(label)
        ax.set_title(f"{TRANSLATIONS[metric]}", fontsize=12)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)
        if metric == 'cost' and not args.hard_constraint:
            threshold = SAFETY_THRESHOLDS.get(args.envs[0], None)
            if threshold is not None:
                ax.axhline(y=threshold, color='red', linestyle='--', label='Safety Threshold')
                ax.text(0.5, threshold, 'Safety Threshold', horizontalalignment='center',
                        verticalalignment='top', transform=ax.get_yaxis_transform(), fontsize=10,
                        style='italic', color='darkred')
        ax.legend()
    plt.tight_layout()
    # Save to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    file = f'{args.method}_{args.envs[0]}_level_{args.level}_{args.inputs[-1].split("/")[-1]}'
    os.makedirs(folder, exist_ok=True)

    # Save both PDF and PNG formats
    pdf_path = f'{folder}/{file}.pdf'
    png_path = f'{folder}/{file}.png'
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    print(f"Plot saved to: {pdf_path}")
    print(f"Plot saved to: {png_path}")
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    # Use the main common parser from results.commons
    parser = create_common_parser("Plot metrics from structured data directory.")

    # Create default path dynamically
    default_main = create_default_paths(__file__, 'main')

    # Override specific arguments for single_env.py
    parser.set_defaults(
        inputs=[default_main],
        method="PPO",
        envs=["armament_burden"],  # Single environment for this script
        seeds=[1, 2, 3]
    )

    # Add script-specific arguments
    parser.add_argument("--total_iterations", type=float, nargs='+', default=[5e8],
                        help="Total number of environment iterations for each input directory")

    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
