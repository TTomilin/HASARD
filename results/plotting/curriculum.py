import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from results.commons import ENV_INITIALS, TRANSLATIONS, load_data, process_metric_data, create_common_parser

TRANSLATIONS['data/main'] = 'Regular'


def process_data(base_paths, method, environments, seeds, metrics, level, n_data_points):
    results = {m: {e: {} for e in environments} for m in metrics}
    for env in environments:
        for base_path in base_paths:
            for metric in metrics:
                stats = process_metric_data(base_path, method, env, seeds, metric, level, n_data_points)
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
    data = process_data(args.inputs, args.method, args.envs, args.seeds, args.metrics, args.level, args.n_data_points)
    plot_results(data, args.inputs, args.envs)


def common_plot_args():
    parser = create_common_parser("Plot reward and cost side by side.")
    parser.set_defaults(
        method="PPOPID",
        level=3,
        seeds=[1, 2, 3],
        envs=["remedy_rush", "collateral_damage"]
    )
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main', 'data/curriculum'])
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
