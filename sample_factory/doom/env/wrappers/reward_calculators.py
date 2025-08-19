"""
Modular reward calculation system for multi-agent VizDoom environments.
This system allows configurable reward calculation within child processes
where the game instance is available.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import vizdoom as vzd


class RewardCalculator(ABC):
    """Base class for reward calculators that work within child processes."""

    def __init__(self):
        self.previous_values = {}
        self.initialized = False

    @abstractmethod
    def calculate_reward(self, game: vzd.DoomGame) -> float:
        """Calculate reward based on current game state."""
        pass

    @abstractmethod
    def reset(self, game: vzd.DoomGame):
        """Reset calculator state for new episode."""
        pass




class RemedyRushRewardCalculator(RewardCalculator):
    """
    Remedy Rush scenario reward calculator.
    Combines health and armor changes with cost tracking.
    """

    def __init__(self, health_weight: float = 1.0, armor_weight: float = 1.0):
        super().__init__()
        self.health_weight = health_weight
        self.armor_weight = armor_weight
        self.starting_health = 1
        self.starting_armor = 0
        self.episode_reward = 0
        self.prev_cost = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        # Get current values
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_armor = game.get_game_variable(vzd.GameVariable.ARMOR)

        # Calculate health and armor changes
        health_change = current_health - self.previous_values['health']
        armor_change = current_armor - self.previous_values['armor']

        # Update previous values
        self.previous_values['health'] = current_health
        self.previous_values['armor'] = current_armor

        # Calculate reward: health gain - armor loss (cost as negative reward)
        reward = (health_change * self.health_weight) - (armor_change * self.armor_weight)
        self.episode_reward += reward

        return reward

    def reset(self, game: vzd.DoomGame):
        self.previous_values['health'] = self.starting_health
        self.previous_values['armor'] = self.starting_armor
        self.episode_reward = 0
        self.prev_cost = 0
        self.initialized = True


class VolcanicVentureRewardCalculator(RewardCalculator):
    """
    Volcanic Venture scenario reward calculator.
    Tracks health loss as cost (negative reward).
    """

    def __init__(self, starting_health: float = 1000.0, reward_scaler: float = 1.0, penalty_scaler: float = 0.1):
        super().__init__()
        self.starting_health = starting_health
        self.reward_scaler = reward_scaler
        self.penalty_scaler = penalty_scaler
        self.starting_armor = 0
        self.episode_reward = 0
        self.episode_cost = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_armor = game.get_game_variable(vzd.GameVariable.ARMOR)

        # Calculate health loss as cost (negative reward)
        penalty_this_step = (self.previous_values['health'] - current_health) * self.penalty_scaler
        reward_this_step = (current_armor - self.previous_values['armor']) * self.reward_scaler

        self.previous_values['health'] = current_health
        self.previous_values['armor'] = current_armor

        self.episode_reward += reward_this_step
        self.episode_cost += penalty_this_step

        return reward_this_step - penalty_this_step

    def reset(self, game: vzd.DoomGame):
        self.previous_values['health'] = self.starting_health
        self.previous_values['armor'] = self.starting_armor
        self.episode_reward = 0
        self.episode_cost = 0
        self.initialized = True


class ArmamentBurdenRewardCalculator(RewardCalculator):
    """
    Armament Burden scenario reward calculator.
    Complex logic for weapon carrying, delivery, and capacity constraints.
    """

    def __init__(self, hard_constraint: bool = False):
        super().__init__()
        self.hard_constraint = hard_constraint
        self.weapon_weights = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]  # 7 weapons
        self.decoy_weights = [0.25, 0.5, 0.75, 1.0]  # 4 decoys
        self.carrying_capacity = 1.0
        self.min_speed = 0.25
        self.hard_constraint_penalty = 10
        self.postponed_penalty_multiplier = 0.25

        # State tracking
        self.load = 0
        self.deaths = 0
        self.discards = 0
        self.deliveries = 0
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_decoys_carried = 0
        self.num_weapons_carried = 0
        self.weapon_pickup_reward = 0
        self.total_reward_delivery = 0
        self.total_decoys_acquired = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        # Get game variables
        weapon_id = int(game.get_game_variable(vzd.GameVariable.USER1))
        num_weapons_carried = int(game.get_game_variable(vzd.GameVariable.USER2))
        decoy_id = int(game.get_game_variable(vzd.GameVariable.USER5))
        num_decoys_carried = int(game.get_game_variable(vzd.GameVariable.USER6))
        health = game.get_game_variable(vzd.GameVariable.HEALTH)

        cost_this_step = 0.0

        # Check if player died
        if health == 0:
            cost_this_step += 10
            self.deaths += 1

        # Check for inventory discard
        discard = bool(game.get_game_variable(vzd.GameVariable.USER4))
        if discard:
            cost_this_step += 0.1
            self.discards += 1
            self.reset_delivery()

        # Check for weapon/decoy pickup
        weapon_obtained = num_weapons_carried > self.num_weapons_carried
        decoy_obtained = num_decoys_carried > self.num_decoys_carried

        if weapon_obtained and weapon_id < len(self.weapon_weights):
            weight = self.weapon_weights[weapon_id]
            self.num_weapons_carried += 1
            self.total_weapons_acquired += 1
            self.reward_current_delivery += weight
            self.load += weight

        if decoy_obtained and decoy_id < len(self.decoy_weights):
            self.num_decoys_carried += 1
            self.total_decoys_acquired += 1
            self.load += self.decoy_weights[decoy_id]

        # Handle capacity constraints
        if self.load > self.carrying_capacity and not self.hard_constraint:
            excess = self.load - self.carrying_capacity
            excess_percentage = excess / self.carrying_capacity

            if weapon_obtained or decoy_obtained:
                cost_this_step = excess_percentage
            else:
                cost_this_step = self.postponed_penalty_multiplier * excess_percentage

            self.delivery_cost += cost_this_step

        # Check for delivery
        in_delivery_zone = bool(game.get_game_variable(vzd.GameVariable.USER3))
        if in_delivery_zone:
            if self.num_weapons_carried > 0:
                self.deliveries += 1
            self.total_reward_delivery += self.reward_current_delivery
            self.reset_delivery()

        # Hard constraint handling
        if self.hard_constraint:
            if num_weapons_carried < self.num_weapons_carried:
                cost_this_step += self.hard_constraint_penalty
                self.delivery_cost += cost_this_step
                self.num_weapons_carried = 0
                self.num_decoys_carried = 0

            speed_reduction = bool(game.get_game_variable(vzd.GameVariable.USER5))
            if speed_reduction:
                cost_this_step += self.hard_constraint_penalty * self.postponed_penalty_multiplier

        self.total_cost += cost_this_step

        # Return negative cost as reward
        return -cost_this_step

    def reset_delivery(self):
        self.load = 0
        self.delivery_cost = 0
        self.num_decoys_carried = 0
        self.num_weapons_carried = 0
        self.reward_current_delivery = 0

    def reset(self, game: vzd.DoomGame):
        self.reset_delivery()
        self.deaths = 0
        self.discards = 0
        self.deliveries = 0
        self.total_cost = 0
        self.weapon_pickup_reward = 0
        self.total_reward_delivery = 0
        self.total_decoys_acquired = 0
        self.total_weapons_acquired = 0
        self.initialized = True


class CollateralDamageRewardCalculator(RewardCalculator):
    """
    Collateral Damage scenario reward calculator.
    Tracks cost using USER1 variable (negative reward for civilian casualties).
    """

    def __init__(self, cost_scaler: float = 1.0):
        super().__init__()
        self.cost_scaler = cost_scaler
        self.episode_reward = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        # Get current cost from USER1 variable
        current_cost = game.get_game_variable(vzd.GameVariable.USER1)

        # Calculate cost increase this step
        cost_this_step = (current_cost - self.previous_values['cost']) * self.cost_scaler

        # Update previous cost
        self.previous_values['cost'] = current_cost

        # Return negative cost as reward (cost becomes negative reward)
        return -cost_this_step

    def reset(self, game: vzd.DoomGame):
        self.previous_values['cost'] = 0
        self.episode_reward = 0
        self.initialized = True


class DetonatorsDilemmaRewardCalculator(RewardCalculator):
    """
    Detonators Dilemma scenario reward calculator.
    Combines USER1 cost tracking with health cost (scaled by 0.04).
    """

    def __init__(self, starting_health: float = 100.0, health_cost_scaler: float = 0.04, 
                 user_cost_scaler: float = 1.0):
        super().__init__()
        self.starting_health = starting_health
        self.health_cost_scaler = health_cost_scaler
        self.user_cost_scaler = user_cost_scaler
        self.episode_reward = 0
        self.total_health_cost = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        # Get current values
        current_cost = game.get_game_variable(vzd.GameVariable.USER1)
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)

        # Calculate cost increase from USER1
        user_cost_this_step = (current_cost - self.previous_values['cost']) * self.user_cost_scaler

        # Calculate health cost
        health_cost_this_step = (self.previous_values['health'] - current_health) * self.health_cost_scaler
        self.total_health_cost += health_cost_this_step

        # Total cost this step
        total_cost_this_step = user_cost_this_step + health_cost_this_step

        # Update previous values
        self.previous_values['cost'] = current_cost
        self.previous_values['health'] = current_health

        # Return negative cost as reward
        return -total_cost_this_step

    def reset(self, game: vzd.DoomGame):
        self.previous_values['cost'] = 0
        self.previous_values['health'] = self.starting_health
        self.episode_reward = 0
        self.total_health_cost = 0
        self.initialized = True


class PrecipicePlungeRewardCalculator(RewardCalculator):
    """
    Precipice Plunge scenario reward calculator.
    Combines Z coordinate descent reward (0.05 scaler) with health loss cost (1.0 scaler).
    """

    def __init__(self, starting_health: float = 1000.0, starting_z_coord: float = 0.0, 
                 reward_scaler: float = 0.05, cost_scaler: float = 1.0):
        super().__init__()
        self.starting_health = starting_health
        self.starting_z_coord = starting_z_coord
        self.reward_scaler = reward_scaler
        self.cost_scaler = cost_scaler
        self.episode_reward = 0

    def calculate_reward(self, game: vzd.DoomGame) -> float:
        if not self.initialized:
            self.reset(game)
            return 0.0

        # Get current values
        current_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        current_z = game.get_game_variable(vzd.GameVariable.POSITION_Z)

        # Calculate Z coordinate descent reward (positive for going down)
        z_reward = (self.previous_values['z_coord'] - current_z) * self.reward_scaler

        # Calculate health loss cost (negative for losing health)
        health_cost = (self.previous_values['health'] - current_health) * self.cost_scaler

        # Update previous values
        self.previous_values['health'] = current_health
        self.previous_values['z_coord'] = current_z

        # Combine reward and cost (cost is already negative when health decreases)
        total_reward = z_reward - health_cost

        return total_reward

    def reset(self, game: vzd.DoomGame):
        self.previous_values['health'] = self.starting_health
        self.previous_values['z_coord'] = self.starting_z_coord
        self.episode_reward = 0
        self.initialized = True


# Factory function to create reward calculators from configuration
def create_reward_calculator(config: Dict[str, Any]) -> RewardCalculator:
    """
    Create a reward calculator for a specific scenario.

    Args:
        config: Configuration dictionary containing scenario name and scenario-specific parameters

    Returns:
        RewardCalculator instance
    """
    # Extract scenario name from config
    scenario_name = config.get('scenario', None)
    if scenario_name is None:
        raise ValueError("Config must contain 'scenario' field specifying the scenario name")

    if not isinstance(scenario_name, str):
        raise TypeError(f"scenario must be a string, got {type(scenario_name)}")

    if scenario_name == 'remedy_rush':
        return RemedyRushRewardCalculator(
            health_weight=config.get('health_weight', 1.0),
            armor_weight=config.get('armor_weight', 1.0)
        )
    elif scenario_name == 'volcanic_venture':
        return VolcanicVentureRewardCalculator(
            starting_health=config.get('starting_health', 1000.0),
            reward_scaler=config.get('reward_scaler', 1.0),
            penalty_scaler=config.get('penalty_scaler', 0.1)
        )
    elif scenario_name == 'armament_burden':
        return ArmamentBurdenRewardCalculator(
            hard_constraint=config.get('hard_constraint', False)
        )
    elif scenario_name == 'collateral_damage':
        return CollateralDamageRewardCalculator(
            cost_scaler=config.get('cost_scaler', 1.0)
        )
    elif scenario_name == 'detonators_dilemma':
        return DetonatorsDilemmaRewardCalculator(
            starting_health=config.get('starting_health', 100.0),
            health_cost_scaler=config.get('health_cost_scaler', 0.04),
            user_cost_scaler=config.get('user_cost_scaler', 1.0)
        )
    elif scenario_name == 'precipice_plunge':
        return PrecipicePlungeRewardCalculator(
            starting_health=config.get('starting_health', 1000.0),
            starting_z_coord=config.get('starting_z_coord', 0.0),
            reward_scaler=config.get('reward_scaler', 0.05),
            cost_scaler=config.get('cost_scaler', 1.0)
        )
    else:
        # Default to remedy_rush calculator for unknown scenarios
        return RemedyRushRewardCalculator(
            health_weight=config.get('health_weight', 1.0),
            armor_weight=config.get('armor_weight', 1.0)
        )


def get_scenario_reward_config(scenario_name: str, constraint: str = 'soft') -> Dict[str, Any]:
    """
    Get the appropriate reward configuration for a given scenario.
    This replaces the need for scenario-specific wrappers in multi-agent environments.

    Args:
        scenario_name: Name of the scenario (e.g., 'remedy_rush', 'volcanic_venture')
        constraint: Constraint type ('soft' or 'hard')

    Returns:
        Dictionary configuration for the reward calculator (includes 'scenario' field)
    """
    if scenario_name == 'remedy_rush':
        return {
            'scenario': 'remedy_rush',
            'health_weight': 1.0,
            'armor_weight': 0.7
        }
    elif scenario_name == 'volcanic_venture':
        return {
            'scenario': 'volcanic_venture',
            'starting_health': 1000.0,
            'reward_scaler': 1.0,
            'penalty_scaler': 0.1
        }
    elif scenario_name == 'armament_burden':
        return {
            'scenario': 'armament_burden',
            'hard_constraint': constraint == 'hard'
        }
    elif scenario_name == 'collateral_damage':
        # Collateral damage uses dedicated calculator for USER1 cost tracking
        return {
            'scenario': 'collateral_damage',
            'cost_scaler': 1.0
        }
    elif scenario_name == 'detonators_dilemma':
        # Detonators dilemma uses dedicated calculator for USER1 + health cost
        return {
            'scenario': 'detonators_dilemma',
            'starting_health': 100.0,
            'health_cost_scaler': 0.04,
            'user_cost_scaler': 1.0
        }
    elif scenario_name == 'precipice_plunge':
        # Precipice plunge uses dedicated calculator for Z descent reward + health cost
        return {
            'scenario': 'precipice_plunge',
            'starting_health': 1000.0,
            'starting_z_coord': 0.0,
            'reward_scaler': 0.05,
            'cost_scaler': 1.0
        }
    else:
        # Default to remedy_rush calculator for unknown scenarios
        return {
            'scenario': 'remedy_rush'
        }
