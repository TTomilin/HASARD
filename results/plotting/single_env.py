import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS
from sample_factory.doom.doom_utils import DOOM_ENVS


def main(args):
    if len(args.total_iterations) != len(args.inputs):
        raise ValueError("The number of total_iterations must match the number of inputs.")
    data = load_data(args.inputs, args.env, args.algo, args.seeds, args.metrics, args.level)
    plot_metrics(data, args)


def load_data(base_paths, environment, method, seeds, metrics, level):
    """Load data from structured directory."""
    data = {}
    for metric in metrics:
        data[metric] = {}
        for base_path in base_paths:
            runs = []
            for seed in seeds:
                file_path = os.path.join(base_path, environment, method, f"level_{level}", f"seed_{seed}", f"{metric}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        runs.append(json.load(file))
                else:
                    print(f"File not found: {file_path}")
            data[metric][base_path] = runs
    return data


def plot_metrics(data, args):
    plt.style.use('seaborn-v0_8-paper')
    num_metrics = len(args.metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(3 * num_metrics, 2.5))
    if num_metrics == 1:
        axs = [axs]
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
                if args.algo == "PPOCost" and metric == "reward":
                    cost_runs = data.get('cost', {}).get(base_path, [])
                    if cost_runs:
                        # Also trim cost_runs to global_min_length
                        cost_runs = [run[:global_min_length] for run in cost_runs]
                        all_costs = np.array(cost_runs)
                        cost_scalar = doom_env_lookup[args.env].penalty_scaling
                        all_runs += all_costs * cost_scalar
                mean = np.mean(all_runs, axis=0)
                ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                num_data_points = all_runs.shape[1]
                total_iterations = float(total_iterations_dict[base_path])
                iterations_per_point = total_iterations / num_data_points
                x = np.arange(num_data_points) * iterations_per_point
                label = TRANSLATIONS.get(base_path, base_path)
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
            threshold = SAFETY_THRESHOLDS.get(args.env, None)
            if threshold is not None:
                ax.axhline(y=threshold, color='red', linestyle='--', label='Safety Threshold')
                ax.text(0.5, threshold, 'Safety Threshold', horizontalalignment='center',
                        verticalalignment='top', transform=ax.get_yaxis_transform(), fontsize=10,
                        style='italic', color='darkred')
        ax.legend()
    plt.tight_layout()
    folder = 'figures'
    file = f'{args.algo}_{args.env}_level_{args.level}_{args.inputs[-1].split("/")[-1]}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{file}.png', dpi=300)
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main'], help="Base input directories containing the data")
    parser.add_argument("--level", type=int, default=1, help="Level of the run(s) to plot")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algo", type=str, default='PPO', choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"],
                        help="Algorithm to download/plot")
    parser.add_argument("--env", type=str, default="armament_burden",
                        choices=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        help="Environment to plot")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'], help="Name of the metrics to download/plot")
    parser.add_argument("--total_iterations", type=float, nargs='+', default=[5e8],
                        help="Total number of environment iterations for each input directory")
    parser.add_argument("--hard_constraint", action='store_true', help="Whether to use hard constraints")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
