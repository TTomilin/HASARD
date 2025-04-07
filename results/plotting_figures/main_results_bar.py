import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS
from sample_factory.doom.env.doom_utils import DOOM_ENVS

# Add your human baseline averages here (reward/cost) for each level and environment:
HUMAN_BASELINES = {
    1: {
        'armament_burden':      {'reward': 18.60, 'cost': 42.83},
        'volcanic_venture':     {'reward': 57.12, 'cost': 45.97},
        'remedy_rush':          {'reward': 51.04, 'cost': 5.65},
        'collateral_damage':    {'reward': 42.33, 'cost': 4.11},
        'precipice_plunge':     {'reward': 243.09,'cost': 46.88},
        'detonators_dilemma':   {'reward': 31.54, 'cost': 4.49},
    },
    2: {
        'armament_burden':      {'reward': 16.85, 'cost': 31.50},
        'volcanic_venture':     {'reward': 42.91, 'cost': 45.09},
        'remedy_rush':          {'reward': 34.72, 'cost': 1.22},
        'collateral_damage':    {'reward': 17.34, 'cost': 3.10},
        'precipice_plunge':     {'reward': 226.11,'cost': 44.35},
        'detonators_dilemma':   {'reward': 34.66, 'cost': 3.09},
    },
    3: {
        'armament_burden':      {'reward': 9.00,   'cost': 31.40},
        'volcanic_venture':     {'reward': 35.27,  'cost': 49.81},
        'remedy_rush':          {'reward': 36.91,  'cost': 4.22},
        'collateral_damage':    {'reward': 16.40,  'cost': 4.99},
        'precipice_plunge':     {'reward': 104.62, 'cost': 21.54},
        'detonators_dilemma':   {'reward': 37.10,  'cost': 4.08},
    },
}

def main(args):
    data = load_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level, args.hard_constraint)

    # Inject Human baseline data as if it's another method
    if "Human" not in args.algos:
        args.algos.append("Human")

    for env in args.envs:
        for metric in args.metrics:
            # We'll treat "Human" as a separate method key
            key = (env, "Human", metric)
            data[key] = []

            # We already have 10-episode averages. Just replicate each value 10 times
            val = HUMAN_BASELINES[args.level][env][metric]
            data[key].append([val]*10)

    plot_bars(data, args)


def load_data(base_path, environments, methods, seeds, metrics, level, hard_constraint):
    data = {}
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    metric_name = f"{metric}_hard" if hard_constraint else metric
                    path = os.path.join(base_path, env, method, f"level_{level}", f"seed_{seed}", f"{metric_name}.json")
                    key = (env, method, metric)
                    if key not in data:
                        data[key] = []
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            data[key].append(json.load(f))
                    else:
                        print(f"File not found: {path}")
    return data


def plot_bars(data, args):
    fig, axs = plt.subplots(3, 4, figsize=(10, 6.5))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.075, hspace=0.25, wspace=0.35)
    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}

    title_axes = [fig.add_subplot(3, 2, i + 1, frame_on=False) for i in range(6)]
    for ax in title_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    method_handles = {}
    for env_i, env in enumerate(args.envs):
        row = env_i // 2
        col_offset = (env_i % 2) * 2
        title_axes[env_i].set_title(TRANSLATIONS[env], fontsize=14)

        for met_i, metric in enumerate(args.metrics):
            ax = axs[row, col_offset + met_i]
            x_pos = np.arange(len(args.algos))
            values_per_method = []

            for method in args.algos:
                key = (env, method, metric)
                if key not in data or not data[key]:
                    values_per_method.append(0.0)
                    continue

                runs = data[key]
                final_values = []
                add_cost_back = (method == "PPOCost" and metric == "reward")
                cost_runs = None
                if add_cost_back:
                    cost_key = (env, method, "cost")
                    if cost_key in data:
                        cost_runs = data[cost_key]
                    else:
                        values_per_method.append(0.0)
                        continue

                for i, run in enumerate(runs):
                    slice_m = run[-10:] if len(run) >= 10 else run
                    if add_cost_back and cost_runs and i < len(cost_runs):
                        slice_c = cost_runs[i][-10:] if len(cost_runs[i]) >= 10 else cost_runs[i]
                        min_len = min(len(slice_m), len(slice_c))
                        cost_scalar = doom_env_lookup[env].penalty_scaling
                        combined = [r + c * cost_scalar for r, c in zip(slice_m[:min_len], slice_c[:min_len])]
                        final_values.extend(combined)
                    else:
                        final_values.extend(slice_m)

                mean_val = np.mean(final_values) if final_values else 0.0
                values_per_method.append(mean_val)

            color_map = plt.get_cmap("tab10")
            colors = [color_map(i % 10) for i in range(len(args.algos))]
            bars = ax.bar(x_pos, values_per_method, align='center', alpha=0.7, color=colors)

            for patch, algo in zip(bars, args.algos):
                if algo not in method_handles:
                    method_handles[algo] = patch

            if metric == 'cost' and not args.hard_constraint:
                thr = SAFETY_THRESHOLDS[env]
                ax.axhline(thr, color='red', linestyle='--')
                ax.text(0, thr, 'Safety\nThreshold', ha='left', va='bottom', color='darkred', style='italic')

            ax.set_xticks([])
            if metric == "reward":
                ylabel = f"{TRANSLATIONS[metric]}↑"
            elif metric == "cost":
                ylabel = f"{TRANSLATIONS[metric]}↓"
            else:
                ylabel = TRANSLATIONS[metric]
            ax.set_ylabel(ylabel, fontsize=12)

    fig.legend(method_handles.values(), method_handles.keys(), loc='lower center',
               ncol=len(args.algos), bbox_to_anchor=(0.5, 0.0), fontsize=11)

    folder = 'figures'
    fname = 'hard_bar' if args.hard_constraint else f'level_{args.level}_bar'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{fname}.pdf', dpi=300)
    plt.show()


def common_plot_args():
    p = argparse.ArgumentParser(description="Plot bars of average final returns/cost.")
    p.add_argument("--input", type=str, default='data/main')
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3])
    p.add_argument("--algos", type=str, nargs='+', default=[
        "PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O", "TRPO", "TRPOLag", "TRPOPID"
    ])
    p.add_argument("--envs", type=str, nargs='+', default=[
        "armament_burden", "volcanic_venture", "remedy_rush",
        "collateral_damage", "precipice_plunge", "detonators_dilemma"
    ])
    p.add_argument("--metrics", type=str, nargs='+', default=["reward", "cost"])
    p.add_argument("--hard_constraint", action='store_true', default=False)
    p.add_argument("--total_iterations", type=int, default=int(5e8))
    return p


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
