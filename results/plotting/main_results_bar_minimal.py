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

from results.commons import TRANSLATIONS, SAFETY_THRESHOLDS, load_full_data, create_default_paths, create_common_parser
from sample_factory.doom.doom_utils import DOOM_ENVS


def reshape_data_if_needed(runs, num_seeds):
    """
    Helper function to reshape flat list data into separate runs per seed.

    Args:
        runs: Either a flat list of floats or a list of lists
        num_seeds: Number of seeds to split the data into

    Returns:
        List of lists, where each inner list represents one seed's data
    """
    # Check if runs is a flat list of floats (due to load_full_data using extend())
    # If so, we need to reshape it back into separate runs per seed
    if runs and isinstance(runs[0], (int, float)):
        # Flat list case - reshape into separate runs
        total_length = len(runs)

        # Only treat as single run if the data is clearly too small to be multi-seed
        # or if we have only one seed requested
        if num_seeds == 1 or total_length < num_seeds * 10:
            runs = [runs]
        else:
            # Try to split into approximately equal chunks
            # Calculate base chunk size and remainder
            base_chunk_size = total_length // num_seeds
            remainder = total_length % num_seeds

            # Split the data, distributing remainder across first few chunks
            new_runs = []
            start_idx = 0
            for i in range(num_seeds):
                # Add one extra element to first 'remainder' chunks
                chunk_size = base_chunk_size + (1 if i < remainder else 0)
                end_idx = start_idx + chunk_size
                new_runs.append(runs[start_idx:end_idx])
                start_idx = end_idx

            runs = new_runs

    # Now handle the case where runs might have different lengths
    if runs and hasattr(runs[0], '__len__'):
        min_length = min(len(run) for run in runs)
        for i in range(len(runs)):
            runs[i] = runs[i][:min_length]

    return runs

# Add your human baseline averages here (reward/cost) for each level and environment:
HUMAN_BASELINES = {
    1: {
        'armament_burden': {'reward': 18.60, 'cost': 42.83},
        'volcanic_venture': {'reward': 57.12, 'cost': 45.97},
        'remedy_rush': {'reward': 51.04, 'cost': 5.65},
        'collateral_damage': {'reward': 42.33, 'cost': 4.11},
        'precipice_plunge': {'reward': 243.09, 'cost': 46.88},
        'detonators_dilemma': {'reward': 31.54, 'cost': 4.49},
    },
    2: {
        'armament_burden': {'reward': 16.85, 'cost': 31.50},
        'volcanic_venture': {'reward': 42.91, 'cost': 45.09},
        'remedy_rush': {'reward': 34.72, 'cost': 1.22},
        'collateral_damage': {'reward': 17.34, 'cost': 3.10},
        'precipice_plunge': {'reward': 226.11, 'cost': 44.35},
        'detonators_dilemma': {'reward': 34.66, 'cost': 3.09},
    },
    3: {
        'armament_burden': {'reward': 9.00, 'cost': 31.40},
        'volcanic_venture': {'reward': 35.27, 'cost': 49.81},
        'remedy_rush': {'reward': 36.91, 'cost': 4.22},
        'collateral_damage': {'reward': 16.40, 'cost': 4.99},
        'precipice_plunge': {'reward': 104.62, 'cost': 21.54},
        'detonators_dilemma': {'reward': 37.10, 'cost': 4.08},
    },
}


def main(args):
    data = load_full_data(args.input, args.envs, args.algos, args.seeds, args.metrics, args.level, args.hard_constraint)

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
            data[key].append([val] * 10)

    plot_bars(data, args)


def plot_bars(data, args):
    n_envs = len(args.envs)
    n_metrics = len(args.metrics)
    fig, axs = plt.subplots(n_envs, n_metrics, figsize=(3 * n_metrics, 2.5 * n_envs))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.15, hspace=0.3, wspace=0.3)
    axs = axs.flatten()

    doom_env_lookup = {spec.name: spec for spec in DOOM_ENVS}

    title_axes = [fig.add_subplot(2, 1, i + 1, frame_on=False) for i in range(n_envs)]
    for ax in title_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    method_handles = {}
    for env_i, env in enumerate(args.envs):
        col_offset = (env_i % 2) * 2
        title_axes[env_i].set_title(TRANSLATIONS[env], fontsize=14)

        for met_i, metric in enumerate(args.metrics):
            ax = axs[col_offset + met_i]
            x_pos = np.arange(len(args.algos))
            values_per_method = []

            for method in args.algos:
                key = (env, method, metric)
                if key not in data or not data[key]:
                    values_per_method.append(0.0)
                    continue

                runs = data[key]
                runs = reshape_data_if_needed(runs, len(args.seeds))
                final_values = []
                add_cost_back = (method == "PPOCost" and metric == "reward")
                cost_runs = None
                if add_cost_back:
                    cost_key = (env, method, "cost")
                    if cost_key in data:
                        cost_runs = data[cost_key]
                        cost_runs = reshape_data_if_needed(cost_runs, len(args.seeds))
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

    # fig.legend(method_handles.values(), method_handles.keys(), loc='lower center',
    #            ncol=len(args.algos), bbox_to_anchor=(0.5, 0.0), fontsize=11)

    fig.legend(method_handles.values(), method_handles.keys(), loc='lower center', ncol=len(args.algos) // 2 + 1,
               bbox_to_anchor=(0.5, 0.0), fontsize=11)

    # Save to results/figures directory (not results/plotting/figures)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    fname = 'hard_bar' if args.hard_constraint else f'level_{args.level}_bar'
    os.makedirs(folder, exist_ok=True)
    full_path = f'{folder}/{fname}_minimal.pdf'
    plt.savefig(full_path, dpi=300)
    print(f"Plot saved to: {full_path}")
    plt.show()


def common_plot_args():
    # Use the main common parser from results.commons
    parser = create_common_parser("Plot bars of average final returns/cost.")

    # Override specific arguments for main results bar minimal
    # Create default path dynamically
    default_main = create_default_paths(__file__, 'main')

    # Override input path default
    parser.set_defaults(input=default_main)

    # Override seeds default to include 3 seeds
    parser.set_defaults(seeds=[1, 2, 3])

    # Add algos argument (different from method in common parser)
    parser.add_argument("--algos", type=str, nargs='+', default=[
        "PPO", "PPOCost", "PPOLag", "PPOSaute", "PPOPID", "P3O",
    ], help="Algorithms to analyze")

    # Override default environments to only use volcanic_venture and remedy_rush
    parser.set_defaults(envs=["volcanic_venture", "remedy_rush"])

    # Add total_iterations argument specific to this script
    parser.add_argument("--total_iterations", type=int, default=int(5e8),
                        help="Total number of environment iterations")

    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
