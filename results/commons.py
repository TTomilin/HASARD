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
    'diff': 'Difference',
    'data/main': 'Default Obs',
    'data/depth': 'Default Obs + Depth Buffer',
    'data/segment': 'Segmentation',
    'data/curriculum': 'Curriculum',
    'data/full_actions': 'Full Actions',
}

ENV_INITIALS = {
    'armament_burden': 'AB',
    'volcanic_venture': 'VV',
    'remedy_rush': 'RR',
    'collateral_damage': 'CD',
    'precipice_plunge': 'PP',
    'detonators_dilemma': 'DD'
}


# Data Loading
def load_data(base_path, method, environment, seed, level, metric_key, ext_name='', ext_value=''):
    level_key = f"level_{level}"
    if ext_name or ext_value:
        level_key += f"/{ext_name}_{ext_value}"
    file_path = os.path.join(base_path, environment, method, level_key, f"seed_{seed}", f"{metric_key}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    else:
        print(f"File not found: {file_path}")
    return None


def load_full_data(base_path, environments, methods, seeds, metrics, level, hard_constraint):
    data = {}
    for env in environments:
        for method in methods:
            for seed in seeds:
                for metric in metrics:
                    key = (env, method, metric)
                    if key not in data:
                        data[key] = []
                    metric_name = f"{metric}_hard" if hard_constraint else metric
                    exp_data = load_data(base_path, method, env, seed, level, metric_name)
                    if exp_data:
                        data[key].append(exp_data)
    return data
