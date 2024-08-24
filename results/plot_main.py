import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

SAFETY_THRESHOLDS = {
    "armament_burden": 50,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 50,
    "detonators_dilemma": 5,
}

TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'precipice_plunge': 'Precipice Plunge',
    'detonators_dilemma': 'Detonator\'s Dilemma',
    'reward': 'Reward',
    'cost': 'Cost',
}


def main(args):
    data = load_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level)
    plot_metrics(data, args)


def load_data(base_path, environments, methods, seeds, metrics, level):
    """Load data from structured directory."""
    data = {}
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    metric_name = f"{metric}_hard" if args.hard_constraint else metric
                    file_path = os.path.join(base_path, env, method, f"level_{level}", f"seed_{seed}", f"{metric_name}.json")
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
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(3, 4, figsize=(12, 8))  # Adjust figsize for better fit

    fig.subplots_adjust(left=0.055, right=0.99, top=0.95, bottom=0.12, hspace=0.5, wspace=0.3)

    lines = []
    labels = []

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
                            all_runs += all_costs  # Modify this line to adjust how cost influences reward

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

    fig.legend(lines, labels, loc='lower center', ncol=len(args.algos), fontsize=12, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.0))

    folder = 'plots'
    file = 'hard' if args.hard_constraint else f'level_{args.level}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{file}.pdf', dpi=300)
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")
    parser.add_argument("--input", type=str, default='data/main', help="Base input directory containing the data")
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
