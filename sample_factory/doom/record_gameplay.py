import sys
from datetime import datetime

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.doom.env.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults
from sample_factory.doom.train_vizdoom import register_vizdoom_components


def main():
    """Script entry point."""
    register_vizdoom_components()
    parser, cfg = parse_sf_args(evaluation=True)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)
    if not cfg.timestamp:
        cfg.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    statuses = []
    for algo in ["PPO", "PPOCost", "PPOLag"]:
        for env in ["armament_burden", "volcanic_venture", "remedy_rush",
                    "collateral_damage", "precipice_plunge", "detonators_dilemma"]:
            for level in [1, 2, 3]:
                cfg.env = env
                cfg.level = level
                cfg.algo = algo
                cfg.episode_horizon = cfg.max_num_frames
                status = enjoy(cfg)
                print(f"{env} Level {level} completed by {algo}. Status: {status}")
                statuses.append(status)
    return status


if __name__ == "__main__":
    sys.exit(main())