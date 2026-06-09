import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import results.commons
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from results.commons import save_plot

def cumulative_frames(time_vals, fps_vals):
    """Compute cumulative frames from time and FPS arrays with zero anchor."""
    t = np.insert(time_vals, 0, [0, 10, 100])
    f = np.insert(fps_vals, 0, [0, 100, 100])
    diffs = np.diff(t, prepend=t[0])
    return t, np.cumsum(f * diffs)


def main(args):
    safety_gym_df = pd.read_csv(args.safety_gym_csv)
    hasard_df = pd.read_csv(args.hasard_csv)
    crax_df = pd.read_csv(args.crax_csv)

    sg_time, sg_cumframes = cumulative_frames(
        safety_gym_df['Time/Total'].values, safety_gym_df['Time/FPS'].values
    )

    hasard_fps = hasard_df['PPO_volcanic_venture_level_1_soft_seed_3_20240810_012623_771642 - perf/_fps'].values
    total_runtime_seconds = 2 * 3600 + 35 * 60 + 21
    hasard_time = np.linspace(0, total_runtime_seconds, len(hasard_fps))
    hasard_cumframes = np.cumsum(hasard_fps * np.diff(hasard_time, prepend=0))

    crax_time, crax_cumframes = cumulative_frames(
        crax_df['Time/Total'].values, crax_df['Time/FPS'].values
    )

    plt.figure(figsize=(5, 4))
    plt.plot(sg_time / 60, sg_cumframes, label='Safety-Gymnasium', color='green')
    plt.plot(hasard_time / 60, hasard_cumframes, label='HASARD', color='blue')
    plt.plot(crax_time / 60, crax_cumframes, label='CRAX', color='orange')

    plt.xlim(-2, 120)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cumulative Frames (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, ls=":")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.dirname(script_dir)
    folder = os.path.join(results_dir, 'figures')
    save_plot('FPS', folder)

    print(f"Average FPS for Safety-Gymnasium: {safety_gym_df['Time/FPS'].mean():.2f}")
    print(f"Average FPS for HASARD: {hasard_fps.mean():.2f}")
    print(f"Average FPS for CRAX: {crax_df['Time/FPS'].mean():.2f}")


def common_plot_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FPS comparison between HASARD, CRAX, and Safety-Gymnasium.")

    default_fps_dir = Path(__file__).parent.parent.resolve() / 'data' / 'fps'

    parser.add_argument("--safety_gym_csv", type=str,
                        default=str(default_fps_dir / 'SafetyPointGoal.csv'),
                        help="Path to Safety-Gymnasium CSV file")
    parser.add_argument("--hasard_csv", type=str,
                        default=str(default_fps_dir / 'VolcanicVenture.csv'),
                        help="Path to HASARD CSV file")
    parser.add_argument("--crax_csv", type=str,
                        default=str(default_fps_dir / 'CRAX.csv'),
                        help="Path to CRAX CSV file")
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
