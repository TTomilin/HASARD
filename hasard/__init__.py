from enum import Enum

import gymnasium
from gymnasium import Env
from gymnasium.envs.registration import register

from hasard.wrappers.cost import CostWrapper

LEVELS = [1, 2, 3]


class Scenario(Enum):
    ARMAMENT_BURDEN = 'ArmamentBurden'
    DETONATORS_DILEMMA = 'DetonatorsDilemma'
    VOLCANIC_VENTURE = 'VolcanicVenture'
    PRECIPICE_PLUNGE = 'PrecipicePlunge'
    COLLATERAL_DAMAGE = 'CollateralDamage'
    REMEDY_RUSH = 'RemedyRush'


def register_environment(scenario, level):
    env_name = f"{scenario.value}Level{level}-v0"
    register(
        id=env_name,
        entry_point=f'hasard.envs.{scenario_enum.name.lower()}.env:{scenario.value}',
        kwargs={'level': level}
    )


# Loop through each scenario, level, and constraint to register environments
for scenario_enum in Scenario:
    for level in LEVELS:
        register_environment(scenario_enum, level)


def make(env_id, **kwargs) -> Env:
    env = gymnasium.make(env_id, **kwargs)
    env = CostWrapper(env)
    return env
