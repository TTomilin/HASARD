import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'reward': 'Reward',
    'avg_cost': 'Cost',
}


def main(args):
    data = load_data(args.input, args.envs, args.algos, args.seeds, args.metrics)
    plot_metrics(data, args)


def load_data(base_path, environments, methods, seeds, metrics):
    """Load data from structured directory."""
    data = {}
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    file_path = os.path.join(base_path, env, method, f"seed_{seed}", f"{metric}.json")
                    key = (env, method, metric)
                    if key not in data:
                        data[key] = []
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as file:
                            data[key].append(json.load(file))
                    else:
                        print(f"File not found: {file_path}")
    return data


def plot_metrics(data, args):
    # print(plt.style.available)
    plt.style.use('seaborn-v0_8-muted')
    for metric in args.metrics:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # Create a 2x2 grid of subplots
        axs = axs.flatten()
        env_mapping = dict(zip(args.envs, axs))
        for env in args.envs:
            ax = env_mapping[env]
            for method in args.algos:
                key = (env, method, metric)
                if key in data and data[key]:
                    try:
                        all_runs = np.array(data[key])
                        if all_runs.size == 0:
                            continue  # Skip if there's no data
                        num_data_points = all_runs.shape[1]  # Get the number of data points from the shape of all_runs
                        iterations_per_point = args.total_iterations / num_data_points  # Calculate iterations per data point dynamically
                        mean = np.mean(all_runs, axis=0)
                        ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                        x = np.arange(len(mean)) * iterations_per_point  # Scale x-values by iterations per data point
                        ax.plot(x, mean, label=method)
                        ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)
                        ax.set_xlim(-args.total_iterations / 60, args.total_iterations)
                        ax.set_ylim(0, None)
                    except Exception as e:
                        print(f"Failed to plot {key}: {e}")
                        continue
            ax.set_title(TRANSLATIONS[env])
            ax.set_xlabel('Environment Iterations')
            ax.set_ylabel(TRANSLATIONS[metric])
            ax.legend()
        plt.tight_layout()
        plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")
    parser.add_argument("--input", type=str, default='data', help="Base input directory containing the data")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag"],
                        help="Algorithms to download/plot")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage"],
                        help="Environments to download/plot")
    parser.add_argument("--metrics", type=str, default=['reward', 'avg_cost'], help="Name of the metrics to download/plot")
    parser.add_argument("--total_iterations", type=int, default=3e8,
                        help="Total number of environment iterations corresponding to 500 data points")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
