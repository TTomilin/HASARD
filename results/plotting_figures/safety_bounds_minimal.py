import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from results.commons import TRANSLATIONS


def main(args):
    data = load_data(args.input, args.envs, args.algo, args.seeds, args.metrics, args.level)
    plot_metrics(data, args)


def load_data(base_path, environments, algo, seeds, metrics, level):
    """Load data from structured directory."""
    data = {}
    for env in environments:
        for seed in seeds:
            for metric in metrics:
                dir_path = os.path.join(base_path, env, algo, f"level_{level}")
                # Read every folder in this directory
                for bound in os.scandir(dir_path):
                    file_path = os.path.join(dir_path, bound.name, f"seed_{seed}", f"{metric}.json")
                    key = (env, bound.name, metric)
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
    n_envs = len(args.envs)
    n_metrics = len(args.metrics)
    fig, axs = plt.subplots(n_envs, n_metrics, figsize=(3 * n_metrics, 2.75 * n_envs))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.95, bottom=0.2, hspace=0.5, wspace=0.3)
    axs = axs.flatten()

    # Dictionary to track lines by label to prevent duplicates
    line_label_dict = {}

    title_axes = [fig.add_subplot(2, 1, i + 1, frame_on=False) for i in range(n_envs)]
    for ax in title_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Collect all bounds to determine the unique colors
    all_bounds = set(key[1] for key in data.keys())
    all_bounds = sorted(all_bounds, key=lambda x: float(x.split('_')[-1]))
    colors = plt.get_cmap('tab20c', len(all_bounds) + 10)

    # Create a mapping from bounds to colors
    bound_colors = {bound: colors(i) for i, bound in enumerate(all_bounds)}

    for env_index, env in enumerate(args.envs):
        col_base = (env_index % 2) * 2
        title_axes[env_index].set_title(TRANSLATIONS[env], fontsize=14)

        for metric_index, metric in enumerate(args.metrics):
            ax = axs[col_base + metric_index]
            for bound in all_bounds:
                bound_label = f"Bound {bound.split('_')[-1]}"
                key = (env, bound, metric)
                if key in data and data[key]:
                    runs = data[key]
                    min_length = min(len(runs[0]), len(runs[1]))
                    runs[0] = runs[0][:min_length]
                    runs[1] = runs[1][:min_length]
                    all_runs = np.array(runs)
                    if all_runs.size == 0 or len(all_runs.shape) < 2:
                        continue

                    num_data_points = all_runs.shape[1]
                    iterations_per_point = args.total_iterations / num_data_points
                    mean = np.mean(all_runs, axis=0)
                    ci = 1.96 * np.std(all_runs, axis=0) / np.sqrt(len(all_runs))
                    x = np.arange(num_data_points) * iterations_per_point
                    line = ax.plot(x, mean, label=bound_label, color=bound_colors[bound])
                    ax.fill_between(x, mean - ci, mean + ci, color=bound_colors[bound], alpha=0.2)
                    ax.set_xlim(-args.total_iterations / 60, args.total_iterations)
                    ax.set_xlabel('Steps', fontsize=12)
                    ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)
                    ax.tick_params(axis='both', labelsize=11)
                    ax.xaxis.get_offset_text().set_fontsize(11)

                    if metric == 'cost':
                        bound_val = float(bound.split('_')[-1])
                        ax.axhline(y=bound_val, color='darkgreen', linestyle=':', label='Safety Threshold')

                    # Manage line-label pairs uniquely
                    if bound_label not in line_label_dict:
                        line_label_dict[bound_label] = line[0]

    # original handles & labels in row‐major order:
    handles = list(line_label_dict.values())
    labels  = list(line_label_dict.keys())

    n = len(labels)
    ncol = len(labels) // 2           # e.g. 2 columns
    nrows = (n + ncol - 1) // ncol    # ceil division

    # allocate arrays for Matplotlib’s column‐major fill
    col_major_handles = [None] * n
    col_major_labels  = [None] * n

    # for each desired row‐major index i, compute where it lands in col‐major:
    for i, (h, lab) in enumerate(zip(handles, labels)):
        row = i // ncol
        col = i % ncol
        new_idx = col * nrows + row
        col_major_handles[new_idx] = h
        col_major_labels[new_idx]  = lab

    fig.legend(col_major_handles, col_major_labels, loc='lower center', ncol=ncol, fontsize=10, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.0))

    folder = 'figures'
    file = f'bounds_{args.algo}_level_{args.level}'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{file}_minimal.pdf', dpi=300)
    plt.show()


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics from structured data directory.")
    parser.add_argument("--input", type=str, default='data/safety_bound',
                        help="Base input directory containing the data")
    parser.add_argument("--level", type=int, default=1, help="Level(s) of the run(s) to plot")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--algo", type=str, default="PPOLag", help="Name of the algorithm")
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["armament_burden", "remedy_rush", "volcanic_venture", "collateral_damage",
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
