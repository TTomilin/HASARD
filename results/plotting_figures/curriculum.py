import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from results.commons import ENV_INITIALS

TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'precipice_plunge': 'Precipice Plunge',
    'detonators_dilemma': "Detonator's Dilemma",
    'reward': 'Reward',
    'cost': 'Cost',
    'data/main': 'Regular',
    'data/curriculum': 'Curriculum'
}


def load_data(base_path, method, environment, seed, level, metric_key):
    file_path = os.path.join(
        base_path, environment, method, f"level_{level}", f"seed_{seed}", f"{metric_key}.json"
    )
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


def process_metric(base_path, method, env, seeds, metric, level, n_data_points):
    metric_vals = []
    for seed in seeds:
        data = load_data(base_path, method, env, seed, level, metric)
        if data and len(data) >= n_data_points:
            if base_path == 'data/main':
                data = data[:300]
            metric_vals.extend(data[-n_data_points:])
    if not metric_vals:
        return {'mean': None, 'ci': None}
    mean_val = np.mean(metric_vals)
    ci_val = 1.96 * np.std(metric_vals) / np.sqrt(len(metric_vals))
    return {'mean': mean_val, 'ci': ci_val}


def process_data(base_paths, method, environments, seeds, metrics, level, n_data_points):
    results = {m: {e: {} for e in environments} for m in metrics}
    for env in environments:
        for base_path in base_paths:
            for metric in metrics:
                stats = process_metric(base_path, method, env, seeds, metric, level, n_data_points)
                results[metric][env][base_path] = stats
    return results


def plot_results(results, base_paths, environments):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    metrics = ['reward', 'cost']

    for ax, metric in zip(axes, metrics):
        x_positions = np.arange(len(environments))
        bar_width = 0.7 / len(base_paths)
        for i, base_path in enumerate(base_paths):
            means = []
            errs = []
            for env in environments:
                data = results[metric][env].get(base_path, {'mean': 0, 'ci': 0})
                means.append(data['mean'] if data['mean'] else 0)
                errs.append(data['ci'] if data['ci'] else 0)
            offset = i * bar_width
            ax.bar(
                x_positions + offset,
                means,
                yerr=errs,
                width=bar_width,
                capsize=5,
                label=TRANSLATIONS[base_path],
            )
        ax.set_ylabel(TRANSLATIONS[metric], fontsize=12)
        ax.set_xticks(x_positions + bar_width * (len(base_paths) - 1) / 2)
        ax.set_xticklabels([ENV_INITIALS.get(env, env) for env in environments])

    # Add dashed red safety threshold lines at y=5 and y=50
    axes[1].axhline(5, color='red', linestyle='--')

    # Create custom legend handles for the safety threshold
    custom_handles = [
        Line2D([0], [0], color='#1f77b4', lw=4, label='Regular'),
        Line2D([0], [0], color='#ff7f0e', lw=4, label='Curriculum'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Safety Threshold')
    ]
    # Increase bottom margin so the legend fits
    fig.legend(handles=custom_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    folder = 'figures'
    os.makedirs(folder, exist_ok=True)
    file_name = f'curriculum_bar'
    plt.savefig(f'{folder}/{file_name}.pdf', dpi=300)
    plt.show()


def main(args):
    data = process_data(
        args.inputs,
        args.method,
        args.envs,
        args.seeds,
        args.metrics,
        args.level,
        args.n_data_points
    )
    plot_results(data, args.inputs, args.envs)


def common_plot_args():
    parser = argparse.ArgumentParser(description="Plot reward and cost side by side.")
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main', 'data/curriculum'])
    parser.add_argument("--method", type=str, default="PPOPID",
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"])
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("--n_data_points", type=int, default=10)
    parser.add_argument("--envs", type=str, nargs='+',
                        default=["remedy_rush", "collateral_damage"],
                        choices=["armament_burden", "volcanic_venture", "remedy_rush",
                                 "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        help="Environments to analyze")
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'])
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
