from sample_factory.enjoy import enjoy, register_vizdoom_components, parse_vizdoom_cfg


def main():
    """Script entry point."""
    register_vizdoom_components()
    cfg = parse_vizdoom_cfg()
    cfg.record_gameplay = True
    cfg.save_video = True
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
    main()
