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

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS, load_full_data, check_data_availability, create_default_paths
from sample_factory.doom.doom_utils import DOOM_ENVS


def main(args):
    # Check if any data is available for the specified path
    if not check_data_availability(args.input, args.algos[0], args.envs, args.seeds, args.metrics, args.level):
        print(f"Error: No data found at the specified path '{args.input}'. Please check that the path contains data for the specified environments, algorithms, seeds, metrics, and level.")
        return

    data = load_full_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level, args.hard_constraint)
    plot_metrics(data, args)


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
                    min_length = min(len(run) for run in runs)
                    for i in range(len(runs)):
                        runs[i] = runs[i][:min_length]
                    all_runs = np.array(runs)

                    if all_runs.size == 0 or len(all_runs.shape) < 2:
                        continue

                    # The reward of PPOCost is logged with the cost subtracted from it
                    # We need to add it back for a fair comparison
                    if method == "PPOCost" and metric == "reward":
                        cost_key = (env, method, "cost")
                        if cost_key in data:
                            all_costs = np.array(data[cost_key])
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

    folder = 'figures'
    file = 'hard' if args.hard_constraint else f'level_{args.level}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{file}.pdf', dpi=300)
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")

    # Create default path dynamically
    default_main = create_default_paths(__file__, 'main')

    parser.add_argument("--input", type=str, default=default_main, help="Base input directory containing the data")
    parser.add_argument("--level", type=int, default=1, help="Level(s) of the run(s) to plot")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O"],
                        help="Algorithms to download/plot")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "volcanic_venture", "remedy_rush", "collateral_damage",
                                 "precipice_plunge", "detonators_dilemma"],
                        help="Environments to download/plot")
    parser.add_argument("--metrics", type=str, default=['reward', 'cost'], help="Name of the metrics to download/plot")
    parser.add_argument('--hard_constraint', default=False, action='store_true', help='Soft/Hard safety constraint')
    parser.add_argument("--total_iterations", type=int, default=5e8,
                        help="Total number of environment iterations corresponding to 500 data points")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
