import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


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
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of subplots
    axs = axs.flatten()
    env_mapping = dict(zip(args.envs, axs))
    for env in args.envs:
        ax = env_mapping[env]
        for method in args.algos:
            for metric in args.metrics:
                key = (env, method, metric)
                if key in data and data[key]:
                    all_runs = np.array(data[key])
                    mean = np.mean(all_runs, axis=0)
                    ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                    x = np.arange(len(mean))
                    ax.plot(x, mean, label=f'{method} - {metric}')
                    ax.fill_between(x, mean - ci, mean + ci, alpha=0.2)
        ax.set_title(env)
        ax.set_xlabel('Steps (thousands)')
        ax.set_ylabel('Value')
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
    parser.add_argument("--metrics", type=str, default=['reward'], help="Name of the metrics to download/plot")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
