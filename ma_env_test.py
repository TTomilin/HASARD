#!/usr/bin/env python3

import os
import time
from random import choice
import vizdoom as vzd
from multiprocessing import Process

def run_host_game_instance(config_file, instance_id, port=5029, steps=500):
    game = vzd.DoomGame()
    game.load_config(config_file)

    game.set_window_visible(True)  # Disable window rendering
    game.set_sound_enabled(False)   # Disable sound

    # Add arguments to host a multiplayer game
    game.add_game_args(f"-host 2 -port {port} -netmode 0 +timelimit 2.0 +sv_spawnfarthest 1")
    game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")

    game.init()
    print(f"Initialized host game instance {instance_id}")

    actions = [
        [True, False, False, False, False],  # MOVE_LEFT
        [False, True, False, False, False],  # MOVE_RIGHT
        [False, False, True, False, False],  # ATTACK
        [False, False, False, True, False],  # MOVE_FORWARD
        [False, False, False, False, True],  # MOVE_BACKWARD
    ]

    for step in range(steps):
        if game.is_episode_finished():
            game.new_episode()
            print(f"Game instance {instance_id} (Host) started a new episode.")

        game.make_action(choice(actions))

        # Optionally, get and print some state information
        state = game.get_state()
        if state:
            print(f"Host Instance {instance_id}, Step {step + 1}, Total Reward: {game.get_total_reward()}")

        time.sleep(0.1)  # Sleep briefly to simulate real-time stepping

    game.close()
    print(f"Host game instance {instance_id} closed.")

def run_join_game_instance(config_file, instance_id, host_address='127.0.0.1', port=5029, steps=500):
    time.sleep(1.0)  # Wait for the host to initialize

    game = vzd.DoomGame()
    game.load_config(config_file)

    game.set_window_visible(True)  # Disable window rendering
    game.set_sound_enabled(False)   # Disable sound

    # Add arguments to join the multiplayer game
    game.add_game_args(f"-join {host_address} -port {port}")
    game.add_game_args(f"+name Player{instance_id} +colorset {instance_id}")

    game.init()
    print(f"Initialized join game instance {instance_id}")

    actions = [
        [True, False, False, False, False],  # MOVE_LEFT
        [False, True, False, False, False],  # MOVE_RIGHT
        [False, False, True, False, False],  # ATTACK
        [False, False, False, True, False],  # MOVE_FORWARD
        [False, False, False, False, True],  # MOVE_BACKWARD
    ]

    for step in range(steps):
        if game.is_episode_finished():
            game.new_episode()
            print(f"Game instance {instance_id} (Joiner) started a new episode.")

        game.make_action(choice(actions))

        # Optionally, get and print some state information
        state = game.get_state()
        if state:
            print(f"Joiner Instance {instance_id}, Step {step + 1}, Total Reward: {game.get_total_reward()}")

        time.sleep(0.1)  # Sleep briefly to simulate real-time stepping

    game.close()
    print(f"Join game instance {instance_id} closed.")

if __name__ == "__main__":
    # Adjust the path to your config file
    config_file = os.path.join(vzd.scenarios_path, "multi.cfg")

    # Create processes for the game instances
    host_process = Process(target=run_host_game_instance, args=(config_file, 1))
    join_process = Process(target=run_join_game_instance, args=(config_file, 2))

    # Start the processes
    host_process.start()
    join_process.start()

    # Wait for the processes to finish
    host_process.join()
    join_process.join()

    print("Game instances have finished.")
