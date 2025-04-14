import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from results.commons import ENV_INITIALS, TRANSLATIONS, load_data

TRANSLATIONS['data/main'] = 'Simplified Actions'


def process_metric(base_path, algo, env, seeds, metric, level, n_data_points, total_iterations):
    recorded_total = 5e8  # originally, data covers 500M steps
    metric_values = []
    for seed in seeds:
        data = load_data(base_path, algo, env, seed, level=level, metric_key=metric)
        if data is None or len(data) == 0:
            continue
        L = len(data)
        iters_per_point = recorded_total / L
        cutoff_index = int(total_iterations / iters_per_point)
        cutoff_index = min(cutoff_index, L)
        if cutoff_index >= n_data_points:
            window = data[cutoff_index - n_data_points:cutoff_index]
        else:
            window = data[:cutoff_index]
        if window:
            metric_values.extend(window)
    if len(metric_values) == 0:
        return {'mean': 0.0, 'ci': 0.0}
    arr = np.array(metric_values)
    mean_val = np.mean(arr)
    ci = 1.96 * np.std(arr) / np.sqrt(len(arr))
    return {'mean': mean_val, 'ci': ci}


def process_data(action_spaces, algo, environments, seeds, metrics, level, n_data_points, total_iterations):
    results = {}
    store = {}
    for env in environments:
        for base_path in action_spaces:
            for metric in metrics:
                out = process_metric(base_path, algo, env, seeds, metric, level, n_data_points, total_iterations)
                store[(env, base_path, metric)] = out['mean']
    for env in environments:
        results[env] = {}
        for metric in metrics:
            simplified = store.get((env, 'data/main', metric), 0.0)
            full = store.get((env, 'data/full_actions', metric), 0.0)
            if simplified != 0.0:
                percent_diff = -((simplified - full) / simplified) * 100
            else:
                percent_diff = 0.0
            results[env][metric] = {
                'data/main': simplified,
                'data/full_actions': full,
                'diff': percent_diff
            }
    return results


def plot_action_space_comparison(results, args):
    envs = args.envs
    n_envs = len(envs)
    reward_simpl = [results[env]['reward']['data/main'] for env in envs]
    reward_full = [results[env]['reward']['data/full_actions'] for env in envs]
    cost_simpl = [results[env]['cost']['data/main'] for env in envs]
    cost_full = [results[env]['cost']['data/full_actions'] for env in envs]

    x_positions = np.arange(n_envs)
    width = 0.4

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(7, 2), tight_layout=True)

    # Reward subplot
    ax_r.bar(x_positions - width / 2, reward_simpl, width, label='Simplified Action Space')
    ax_r.bar(x_positions + width / 2, reward_full, width, label='Original Action Space')
    ax_r.set_ylabel("Reward")
    ax_r.set_xticks(x_positions)
    ax_r.set_xticklabels([ENV_INITIALS.get(env, env) for env in envs])

    # Cost subplot
    ax_c.bar(x_positions - width / 2, cost_simpl, width, label='Simplified Action Space')
    ax_c.bar(x_positions + width / 2, cost_full, width, label='Original Action Space')
    ax_c.set_ylabel("Cost")
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels([ENV_INITIALS.get(env, env) for env in envs])

    # Add dashed red safety threshold lines at y=5 and y=50
    ax_c.axhline(5, color='red', linestyle='--')
    ax_c.axhline(50, color='red', linestyle='--')

    # Create custom legend handles for the safety threshold
    custom_handles = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Simplified Action Space'),
        Line2D([0], [0], color='#ff7f0e', lw=4, label='Original Action Space'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Safety Threshold')
    ]
    # Increase bottom margin so the legend fits
    fig.legend(handles=custom_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    folder = 'figures'
    os.makedirs(folder, exist_ok=True)
    file_name = f'actions_level_{args.level}_bar'
    plt.savefig(f'{folder}/{file_name}.pdf', dpi=300)
    plt.show()


def main(args):
    results = process_data(args.inputs, args.algo, args.envs, args.seeds, args.metrics, args.level, args.n_data_points,
                           args.total_iterations)
    plot_action_space_comparison(results, args)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot bar charts for reward & cost comparing simplified vs. original action spaces."
    )
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main', 'data/full_actions'],
                        help="Directories: 'data/main' = Simplified, 'data/full_actions' = Original.")
    parser.add_argument("--level", type=int, default=1, help="Which level to plot.")
    parser.add_argument("--algo", type=str, default="PPOLag", help="Name of the algorithm")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3],
                        help="Which seeds to include.")
    parser.add_argument("--n_data_points", type=int, default=10,
                        help="Number of data points to average at the cutoff.")
    parser.add_argument("--total_iterations", type=float, default=2e8,
                        help="Data cutoff in environment steps (e.g., 200M steps).")
    parser.add_argument("--envs", type=str, nargs='+', default=[
        "armament_burden", "volcanic_venture", "remedy_rush",
        "collateral_damage", "precipice_plunge", "detonators_dilemma"
    ], help="List of environments to compare.")
    parser.add_argument("--metrics", type=str, nargs='+', default=["reward", "cost"],
                        help="Which metrics to compare.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
