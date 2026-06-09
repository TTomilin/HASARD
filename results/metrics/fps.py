import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

DEFAULT_FPS_DIR = Path(__file__).parent.parent.resolve() / 'data' / 'fps'

HASARD_FPS_COL = 'PPO_volcanic_venture_level_1_soft_seed_3_20240810_012623_771642 - perf/_fps'
HASARD_RUNTIME_S = 2 * 3600 + 35 * 60 + 21
TWO_HOURS_S = 2 * 3600


def parse_log_updates(log_path):
    ts_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]'
    ver_pattern = r'policy_version (\d+)'
    timestamps, versions = [], []
    with open(log_path) as f:
        for line in f:
            if 'Updated weights for policy' in line:
                m_ts = re.search(ts_pattern, line)
                m_ver = re.search(ver_pattern, line)
                if m_ts and m_ver:
                    ts = datetime.datetime.strptime(m_ts.group(0).strip('[]'), '%Y-%m-%d %H:%M:%S,%f')
                    timestamps.append(ts)
                    versions.append(int(m_ver.group(1)))
    return timestamps, versions


def hasard_fps_stats(csv_path):
    df = pd.read_csv(csv_path)
    fps = df[HASARD_FPS_COL].values
    time = np.linspace(0, HASARD_RUNTIME_S, len(fps))
    cumframes = np.cumsum(fps * np.diff(time, prepend=0))
    # Scale cumulative frames to exactly 2h for fair comparison
    two_h_idx = np.searchsorted(time, TWO_HOURS_S)
    frames_2h = cumframes[min(two_h_idx, len(cumframes) - 1)]
    return fps.mean(), frames_2h


def hasard_update_stats(log_path):
    # Each log line = learner published updated weights after one mini-batch gradient step.
    # This is the async equivalent of one outer PPO iteration.
    timestamps, _ = parse_log_updates(log_path)
    if not timestamps:
        return None, None
    start = timestamps[0]
    elapsed = [(ts - start).total_seconds() for ts in timestamps]
    within_2h = [t for t in elapsed if t <= TWO_HOURS_S]
    total_updates = len(within_2h)
    duration = within_2h[-1] if len(within_2h) > 1 else 1
    updates_per_min = total_updates / (duration / 60)
    return updates_per_min, total_updates


def csv_fps_stats(df):
    fps = df['Time/FPS'].values
    time = df['Time/Total'].values
    cumframes = np.cumsum(fps * np.diff(time, prepend=time[0]))
    frames_2h = fps.mean() * TWO_HOURS_S
    return fps.mean(), frames_2h


def csv_update_stats(df):
    avg_epoch_time = df['Time/Update'].mean()
    updates_per_min = 60 / avg_epoch_time
    total_updates = TWO_HOURS_S / avg_epoch_time
    return updates_per_min, total_updates


def fmt_fps(v):
    if v >= 1_000_000:
        return f'{v/1_000_000:.2f}M'
    if v >= 1_000:
        return f'{v/1_000:.1f}K'
    return f'{v:.1f}'


def fmt_frames(v):
    if v >= 1e9:
        return f'{v/1e9:.2f}B'
    if v >= 1e6:
        return f'{v/1e6:.2f}M'
    if v >= 1e3:
        return f'{v/1e3:.1f}K'
    return f'{v:.0f}'


def main(args):
    rows = []

    # HASARD
    if os.path.exists(args.hasard_csv) and os.path.exists(args.hasard_log):
        avg_fps, frames_2h = hasard_fps_stats(args.hasard_csv)
        upd_per_min, total_upd = hasard_update_stats(args.hasard_log)
        rows.append(('HASARD', avg_fps, frames_2h, upd_per_min, total_upd))
    else:
        print(f"Warning: HASARD files not found, skipping.")

    # Safety-Gymnasium
    if os.path.exists(args.safety_gym_csv):
        sg_df = pd.read_csv(args.safety_gym_csv)
        avg_fps, frames_2h = csv_fps_stats(sg_df)
        upd_per_min, total_upd = csv_update_stats(sg_df)
        rows.append(('Safety-Gymnasium', avg_fps, frames_2h, upd_per_min, total_upd))
    else:
        print(f"Warning: {args.safety_gym_csv} not found, skipping.")

    # CRAX
    if os.path.exists(args.crax_csv):
        crax_df = pd.read_csv(args.crax_csv)
        avg_fps, frames_2h = csv_fps_stats(crax_df)
        upd_per_min, total_upd = csv_update_stats(crax_df)
        rows.append(('CRAX', avg_fps, frames_2h, upd_per_min, total_upd))
    else:
        print(f"Warning: {args.crax_csv} not found, skipping.")

    if not rows:
        print("No data to display.")
        return

    # Print tables
    col_w = [20, 12, 14, 14, 14]
    headers = ['Framework', 'Avg FPS', 'Frames (2h)', 'Updates/min', 'Updates (2h)']
    sep = '+' + '+'.join('-' * w for w in col_w) + '+'

    print('\n=== Throughput & Policy Update Comparison ===\n')
    print(sep)
    print('|' + '|'.join(h.center(w) for h, w in zip(headers, col_w)) + '|')
    print(sep)
    for name, avg_fps, frames_2h, upd_per_min, total_upd in rows:
        upd_min_str = f'{upd_per_min:.1f}' if upd_per_min is not None else 'N/A'
        upd_total_str = fmt_frames(total_upd) if total_upd is not None else 'N/A'
        cols = [name, fmt_fps(avg_fps), fmt_frames(frames_2h), upd_min_str, upd_total_str]
        print('|' + '|'.join(c.center(w) for c, w in zip(cols, col_w)) + '|')
    print(sep)
    print()


def common_plot_args():
    parser = argparse.ArgumentParser(description="Print FPS and policy update comparison table.")
    parser.add_argument("--hasard_csv", type=str,
                        default=str(DEFAULT_FPS_DIR / 'VolcanicVenture.csv'))
    parser.add_argument("--hasard_log", type=str,
                        default=str(DEFAULT_FPS_DIR / 'VolcanicVenture_Updates.out'))
    parser.add_argument("--safety_gym_csv", type=str,
                        default=str(DEFAULT_FPS_DIR / 'SafetyPointGoal.csv'))
    parser.add_argument("--crax_csv", type=str,
                        default=str(DEFAULT_FPS_DIR / 'CRAX.csv'))
    return parser


if __name__ == "__main__":
    parser = common_plot_args()
    args = parser.parse_args()
    main(args)
