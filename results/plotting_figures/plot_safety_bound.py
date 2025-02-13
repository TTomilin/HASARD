import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from results.commons import TRANSLATIONS


def main(args):
    data = load_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level)
    plot_metrics(data, args)


def load_data(base_path, environments, algos, seeds, metrics, level):
    """Load data from structured directory."""
    data = {}
    for env in environments:
        for algo in algos:
            for seed in seeds:
                for metric in metrics:
                    dir_path = os.path.join(base_path, env, algo, f"level_{level}")
                    # Read every folder in this directory
                    for bound in os.scandir(dir_path):
                        file_path = os.path.join(dir_path, bound.name, f"seed_{seed}", f"{metric}.json")
                        key = (env, algo, bound.name, metric)
                        if key not in data:
                            data[key] = []
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as file:
                                data[key].append(json.load(file))
                        else:
                            print(f"File not found: {file_path}")
    return data


def plot_metrics(data, args):
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # Adjust figsize for better fit

    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.12, hspace=0.5, wspace=0.3)

    colors = plt.get_cmap('tab10')
    line_styles = ['-', '--', '-.', (0, (3, 5, 1, 5)),
                   (0, (3, 1, 1, 1))]  # Different line styles for different bounds

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

        algo_color = {}
        for algo_index, algo in enumerate(args.algos):
            algo_color[algo] = colors(algo_index)

        bounds = set()
        for (e, a, b, m), values in data.items():
            if e == env:
                bounds.add(b)

        sorted_bounds = sorted(bounds)  # Sort bounds for consistent order

        for bound_index, bound in enumerate(sorted_bounds):
            for algo_index, algo in enumerate(args.algos):
                for metric_index, metric in enumerate(args.metrics):
                    ax = axs[row, col_base + metric_index]
                    bound_label = f"Bound {bound.split('_')[-1]}"
                    key = (env, algo, bound, metric)
                    if key in data and data[key]:
                        runs = data[key]
                        min_length = min(len(runs[0]), len(runs[1]))
                        runs[0] = runs[0][:min_length]
                        runs[1] = runs[1][:min_length]
                        all_runs = np.array(runs)
                        if all_runs.size == 0 or len(all_runs.shape) < 2:
                            continue

                        # The reward of PPOCost is logged with the cost subtracted from it
                        # We need to add it back for a fair comparison
                        if metric == "reward":
                            cost_key = (env, algo, bound, "cost")
                            if cost_key in data:
                                run_cost = data[cost_key]
                                min_length = min(len(run_cost[0]), len(run_cost[1]))
                                run_cost[0] = run_cost[0][:min_length]
                                run_cost[1] = run_cost[1][:min_length]
                                all_costs = np.array(run_cost)
                                all_runs += all_costs  # Modify this line to adjust how cost influences reward

                        num_data_points = all_runs.shape[1]
                        iterations_per_point = args.total_iterations / num_data_points
                        mean = np.mean(all_runs, axis=0)
                        ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                        x = np.arange(num_data_points) * iterations_per_point
                        line = ax.plot(x, mean, label=bound_label, color=algo_color[algo],
                                       linestyle=line_styles[bound_index % len(line_styles)])
                        ax.fill_between(x, mean - ci, mean + ci, color=algo_color[algo], alpha=0.2)
                        ax.set_xlim(-args.total_iterations / 60, args.total_iterations)

                        y_lim = None
                        # if env == 'armament_burden':
                        #     y_lim = 19 if metric == 'reward' else 11

                        ax.set_ylim(0, y_lim)
                        ax.set_xlabel('Steps', fontsize=12)
                        ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)
                        if env_index == 1 and metric_index == 1:  # Adjust if needed
                            lines.append(line[0])
                            labels.append(bound_label)
                    # Set the legend inside each subplot
                    ax.legend(loc='upper right', fontsize=10)
    # fig.legend(lines, labels, loc='lower center', ncol=len(args.scales), fontsize=12, fancybox=True, shadow=True,
    #            bbox_to_anchor=(0.5, 0.0))

    folder = 'figures'
    file = f'safety_bound_level_{args.level}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{file}.pdf', dpi=300)
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")
    parser.add_argument("--input", type=str, default='data/safety_bound',
                        help="Base input directory containing the data")
    parser.add_argument("--level", type=int, default=1, help="Level(s) of the run(s) to plot")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2], help="Seed(s) of the run(s) to plot")
    # parser.add_argument("--scales", type=float, nargs='+', default=[0.1, 0.5, 1, 2], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algos", type=str, nargs='+', default=["PPOLag", "PPOSaute"], help="Name of the algorithm")
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
