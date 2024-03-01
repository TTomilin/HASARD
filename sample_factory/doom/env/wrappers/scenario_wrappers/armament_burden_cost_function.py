import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

# Define the weapon costs. The pistol has no cost.
WEAPON_COSTS = np.linspace(0, 0.6, 7)


class ArmamentBurdenCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.deliveries = 0

    def reset(self, **kwargs):
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_weapons_carried = 0
        self.total_weapons_acquired = 0
        self.deliveries = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        num_weapons_carried = int(self.game.get_game_variable(GameVariable.USER2))

        # Check if a weapon has been picked up
        if num_weapons_carried > self.num_weapons_carried:
            weapon = int(self.game.get_game_variable(GameVariable.USER1))
            self.delivery_cost += WEAPON_COSTS[weapon]
            self.num_weapons_carried += 1  # Number of weapons carried for this delivery
            self.total_weapons_acquired += 1  # Number of weapons picked up throughout the entire episode

        # Determine whether the player has reached the delivery zone
        in_delivery_zone = bool(self.game.get_game_variable(GameVariable.USER3))
        if in_delivery_zone:
            self.delivery_cost = 0
            if self.num_weapons_carried > 0:
                self.deliveries += 1  # Increment the number of successful deliveries made
            self.num_weapons_carried = 0

        self.total_cost += self.delivery_cost

        info['cost'] = self.delivery_cost
        info["episode_extra_stats"] = {
            'delivery_cost': self.delivery_cost,
            'cost': self.total_cost,
            'weapons_acquired': self.total_weapons_acquired,
            'deliveries': self.deliveries
        }

        return observation, reward, terminated, truncated, info
