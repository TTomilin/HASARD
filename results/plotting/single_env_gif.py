import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import SAFETY_THRESHOLDS, TRANSLATIONS, create_default_paths, create_common_parser
from sample_factory.doom.doom_utils import DOOM_ENVS


def main(args):
    if len(args.num_datapoints) != len(args.inputs):
        raise ValueError("The number of num_datapoints must match the number of inputs.")
    data = load_data(args.inputs, args.env, args.method, args.seeds, args.metrics, args.level)
    animate_metrics(data, args)


def load_data(base_paths, environment, method, seeds, metrics, level):
    data = {}
    for metric in metrics:
        data[metric] = {}
        for base_path in base_paths:
            runs = []
            for seed in seeds:
                file_path = os.path.join(base_path, environment, method,
                                         f"level_{level}", f"seed_{seed}", f"{metric}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        runs.append(json.load(file))
                else:
                    print(f"File not found: {file_path}")
            data[metric][base_path] = runs
    return data


def animate_metrics(data, args):
    plt.style.use('seaborn-v0_8-paper')
    num_metrics = len(args.metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 2.5))
    fig.subplots_adjust(bottom=0.2)
    if num_metrics == 1:
        axs = [axs]

    dp_dict = dict(zip(args.inputs, args.num_datapoints))
    steps_per_dp = args.steps_per_dp

    # Precompute plot data and global y-limits.
    plot_data = {metric: {} for metric in args.metrics}
    global_frames = {}
    global_ymax = {m: 0 for m in args.metrics}
    for metric in args.metrics:
        frames_list = []
        for base_path in args.inputs:
            runs = data[metric].get(base_path, [])
            if not runs:
                continue
            desired_points = int(dp_dict[base_path])
            total_steps = desired_points * steps_per_dp
            sliced_runs = []
            for run in runs:
                if len(run) >= 1:
                    sliced_runs.append(np.array(run[:min(len(run), desired_points)]))
            if not sliced_runs:
                continue
            effective_len = min(r.shape[0] for r in sliced_runs)
            sliced_runs = [r[:effective_len] for r in sliced_runs]
            arr = np.stack(sliced_runs, axis=0)
            if args.method == "PPOCost" and metric == "reward":
                cost_runs = data.get('cost', {}).get(base_path, [])
                if cost_runs:
                    cost_sliced = []
                    for r in cost_runs:
                        if len(r) >= 1:
                            cost_sliced.append(np.array(r[:min(len(r), desired_points)]))
                    if cost_sliced:
                        cost_effective = min(r.shape[0] for r in cost_sliced)
                        cost_sliced = [r[:cost_effective] for r in cost_sliced]
                        cost_arr = np.stack(cost_sliced, axis=0)
                        common_len = min(effective_len, cost_arr.shape[1])
                        arr = arr[:, :common_len] + cost_arr[:, :common_len] * DOOM_ENVS[args.env].penalty_scaling
                        effective_len = common_len
            mean = np.mean(arr, axis=0)
            ci = 1.96 * np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            x_vals = np.linspace(0, total_steps, effective_len)
            plot_data[metric][base_path] = (x_vals, mean, ci, effective_len)
            frames_list.append(effective_len)
            global_ymax[metric] = max(global_ymax[metric], np.max(mean + ci))
        global_frames[metric] = max(frames_list) if frames_list else 0

    if not any(global_frames.values()):
        print("No data available for animation.")
        return

    lines_info = {metric: {} for metric in args.metrics}
    max_steps = max(dp * steps_per_dp for dp in args.num_datapoints)

    def init():
        for i, metric in enumerate(args.metrics):
            ax = axs[i]
            ax.clear()
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel(TRANSLATIONS.get(metric, metric), fontsize=12)
            if metric == 'cost' and not args.hard_constraint:
                thr = SAFETY_THRESHOLDS.get(args.env, None)
                if thr is not None:
                    ax.axhline(y=thr, color='red', linestyle='--', label='Safety Threshold')

            # Fix y-axis based on overall max.
            ax.set_ylim(0, global_ymax[metric] * 1.1)
            # Set fixed x-axis limit if not shifting.
            if not args.shift_x_axis:
                ax.set_xlim(0, max_steps)

            for base_path in plot_data[metric]:
                line, = ax.plot([], [], label=TRANSLATIONS.get(base_path, base_path))
                lines_info[metric][base_path] = line

            if args.plot_legend:
                ax.legend()
        return []

    def update(frame):
        for i, metric in enumerate(args.metrics):
            ax = axs[i]
            # Remove previous fill_between collections.
            for coll in list(ax.collections):
                coll.remove()

            for base_path, line in lines_info[metric].items():
                x_vals, mean, ci, eff_len = plot_data[metric][base_path]
                current = min(frame, eff_len - 1)
                revealed_mean = np.full_like(mean, np.nan)
                revealed_mean[:current + 1] = mean[:current + 1]
                line.set_data(x_vals, revealed_mean)
                color = line.get_color()

                revealed_lower = np.full_like(mean, np.nan)
                revealed_upper = np.full_like(mean, np.nan)
                revealed_lower[:current + 1] = (mean - ci)[:current + 1]
                revealed_upper[:current + 1] = (mean + ci)[:current + 1]
                ax.fill_between(x_vals, revealed_lower, revealed_upper, facecolor=color, alpha=0.2)

            # Update x-axis only if shifting is enabled.
            if args.shift_x_axis:
                # Get current x-value from any dataset.
                x_vals_any = next(iter(plot_data[metric].values()))[0]
                current_x = x_vals_any[min(frame, len(x_vals_any) - 1)]
                ax.set_xlim(0, current_x)
        return []

    max_frames = max(global_frames.values())
    anim = FuncAnimation(fig, update, frames=range(max_frames), init_func=init,
                         blit=False, interval=10)

    # Save to results/figures/animated directory (not results/plotting/figures/animated)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures', 'animated')
    os.makedirs(folder, exist_ok=True)
    metrics_str = f'_{args.metrics[0]}' if len(args.metrics) == 1 else ''
    shift_str = f'_shifting' if args.shift_x_axis else ''
    file_name = f'{args.method}_{args.env}_level_{args.level}{metrics_str}{shift_str}'

    # Save animated GIF
    gif_path = os.path.join(folder, f'{file_name}.gif')
    anim.save(gif_path, writer='pillow', fps=args.fps, dpi=300)
    print(f"GIF saved to: {gif_path}")
    plt.show()


def common_plot_args():
    parser = create_common_parser(
        "Animate metrics from structured data, using a fixed y-axis and configurable x-axis behavior."
    )

    # Create default path dynamically
    default_main = create_default_paths(__file__, 'main')

    # Set defaults for common arguments
    parser.set_defaults(
        inputs=[default_main],
        level=1,
        seeds=[1, 2, 3],
        method='PPO'
    )

    # Add animation-specific arguments
    parser.add_argument("--env", type=str, default="armament_burden",
                        choices=["armament_burden", "volcanic_venture", "remedy_rush",
                                 "collateral_damage", "precipice_plunge", "detonators_dilemma"],
                        help="Single environment to animate (overrides --envs)")
    parser.add_argument("--num_datapoints", type=float, nargs='+', default=[100],
                        help="Number of datapoints per input directory to consider (each datapoint represents multiple steps)")
    parser.add_argument("--steps_per_dp", type=float, default=1e6,
                        help="Number of environment steps represented by one datapoint")
    parser.add_argument("--fps", type=float, default=20, help="Number of datapoints to animate per second")
    parser.add_argument("--plot_legend", action='store_true')
    parser.add_argument("--shift_x_axis", action='store_true',
                        help="Shift the x-axis with new data (otherwise remains fixed)")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
