import json
import os

SAFETY_THRESHOLDS = {
    "armament_burden": 50,
    "volcanic_venture": 50,
    "remedy_rush": 5,
    "collateral_damage": 5,
    "precipice_plunge": 50,
    "detonators_dilemma": 5,
}

TRANSLATIONS = {
    'armament_burden': 'Armament Burden',
    'volcanic_venture': 'Volcanic Venture',
    'remedy_rush': 'Remedy Rush',
    'collateral_damage': 'Collateral Damage',
    'precipice_plunge': 'Precipice Plunge',
    'detonators_dilemma': 'Detonator\'s Dilemma',
    'reward': 'Reward',
    'cost': 'Cost',
    'data/main': 'Default Obs',
    'data/depth': 'Default Obs + Depth Buffer',
    'data/segment': 'Segmentation',
}

ENV_INITIALS = {
    'armament_burden': 'AB',
    'volcanic_venture': 'VV',
    'remedy_rush': 'RR',
    'collateral_damage': 'CD',
    'precipice_plunge': 'PP',
    'detonators_dilemma': 'DD'
}


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
