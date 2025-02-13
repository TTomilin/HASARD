import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from results.commons import SAFETY_THRESHOLDS, TRANSLATIONS
from sample_factory.doom.env.doom_utils import DOOM_ENVS


def main(args):
    if len(args.num_datapoints) != len(args.inputs):
        raise ValueError("The number of num_datapoints must match the number of inputs.")
    data = load_data(args.inputs, args.env, args.algo, args.seeds, args.metrics, args.level)
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
    if num_metrics == 1:
        axs = [axs]
    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}

    # dp_dict holds the number of datapoints to consider for each input.
    dp_dict = dict(zip(args.inputs, args.num_datapoints))
    steps_per_dp = args.steps_per_dp  # new multiplier: e.g. 1e6 steps per datapoint

    # Precompute plot data per metric and base_path.
    # For each, store (x_vals, mean, ci, effective_length)
    plot_data = {metric: {} for metric in args.metrics}
    global_frames = {}
    for metric in args.metrics:
        frames_list = []
        for base_path in args.inputs:
            runs = data[metric].get(base_path, [])
            if not runs:
                continue
            desired_points = int(dp_dict[base_path])  # use only this many datapoints
            total_steps = desired_points * steps_per_dp  # x_max value
            sliced_runs = []
            for run in runs:
                if len(run) >= 1:
                    sliced_runs.append(np.array(run[:min(len(run), desired_points)]))
            if not sliced_runs:
                continue
            effective_len = min(r.shape[0] for r in sliced_runs)
            sliced_runs = [r[:effective_len] for r in sliced_runs]
            arr = np.stack(sliced_runs, axis=0)
            if args.algo == "PPOCost" and metric == "reward":
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
                        arr = arr[:, :common_len] + cost_arr[:, :common_len] * doom_env_lookup[args.env].penalty_scaling
                        effective_len = common_len
            mean = np.mean(arr, axis=0)
            ci = 1.96 * np.std(arr, axis=0) / np.sqrt(arr.shape[0])
            # x_vals now spans 0 to total_steps even if effective_len < desired_points
            x_vals = np.linspace(0, total_steps, effective_len)
            plot_data[metric][base_path] = (x_vals, mean, ci, effective_len)
            frames_list.append(effective_len)
        global_frames[metric] = max(frames_list) if frames_list else 0

    if not any(global_frames.values()):
        print("No data available for animation.")
        return

    # Prepare line objects storage.
    lines_info = {metric: {} for metric in args.metrics}

    def init():
        for i, metric in enumerate(args.metrics):
            ax = axs[i]
            ax.clear()
            ax.set_title(TRANSLATIONS.get(metric, metric), fontsize=12)
            ax.set_xlabel('Steps', fontsize=12)
            ax.set_ylabel(TRANSLATIONS.get(metric, metric), fontsize=12)
            if metric == 'cost' and not args.hard_constraint:
                thr = SAFETY_THRESHOLDS.get(args.env, None)
                if thr is not None:
                    ax.axhline(y=thr, color='red', linestyle='--', label='Safety Threshold')
            # Set x-limit as the maximum steps among inputs.
            max_steps = max(dp * steps_per_dp for dp in args.num_datapoints)
            ax.set_xlim(0, max_steps)
            ax.set_ylim(0, None)
            for base_path in plot_data[metric]:
                line, = ax.plot([], [], label=TRANSLATIONS.get(base_path, base_path))
                lines_info[metric][base_path] = line
            if args.plot_legend:
                ax.legend()
        return []

    def update(frame):
        for i, metric in enumerate(args.metrics):
            ax = axs[i]
            for coll in list(ax.collections):
                coll.remove()
            for base_path, line in lines_info[metric].items():
                x_vals, mean, ci, eff_len = plot_data[metric][base_path]
                current = frame if frame < eff_len else eff_len
                line.set_data(x_vals[:current], mean[:current])
                ax.fill_between(x_vals[:current],
                                (mean - ci)[:current],
                                (mean + ci)[:current],
                                alpha=0.2)
            max_steps = max(dp * steps_per_dp for dp in args.num_datapoints)
            ax.set_xlim(0, max_steps)
            all_y = []
            for base_path in plot_data[metric]:
                _, m, _, eff_len = plot_data[metric][base_path]
                current = frame if frame < eff_len else eff_len
                all_y.extend(m[:current])
            if all_y:
                ax.set_ylim(0, max(all_y) * 1.1)
        return []

    max_frames = max(global_frames.values())
    skip = 1
    anim = FuncAnimation(fig, update, frames=range(0, max_frames, skip), init_func=init, blit=False, interval=10)
    folder = 'figures'
    os.makedirs(folder, exist_ok=True)
    file_name = f'{args.algo}_{args.env}_level_{args.level}_{args.inputs[-1].split("/")[-1]}_animated.gif'
    anim.save(os.path.join(folder, file_name), writer='pillow', fps=20)
    plt.show()


def common_plot_args():
    parser = argparse.ArgumentParser(
        description="Animate metrics from structured data, "
                    "only considering data from index 0 to num_datapoints (each representing several env steps)."
    )
    parser.add_argument("--inputs", type=str, nargs='+', default=['data/main'],
                        help="Base input directories")
    parser.add_argument("--level", type=int, default=1, help="Level")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3],
                        help="Seeds")
    parser.add_argument("--algo", type=str, default='PPO',
                        choices=["PPO", "PPOCost", "PPOLag", "PPOSaute",
                                 "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"])
    parser.add_argument("--env", type=str, default="armament_burden",
                        choices=["armament_burden", "volcanic_venture", "remedy_rush",
                                 "collateral_damage", "precipice_plunge", "detonators_dilemma"])
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'cost'])
    parser.add_argument("--num_datapoints", type=float, nargs='+', default=[100],
                        help="Number of datapoints per input directory to consider (each datapoint represents multiple steps)")
    parser.add_argument("--steps_per_dp", type=float, default=1e6,
                        help="Number of environment steps represented by one datapoint")
    parser.add_argument("--hard_constraint", action='store_true')
    parser.add_argument("--plot_legend", action='store_true')
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
