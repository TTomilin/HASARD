import hasard

def run_task(scenario: hasard.Scenario, level: int, max_steps: int = 250):
    """
    Runs a single task for the specified environment and level, capped at max_steps.

    Args:
        scenario (Scenario): Scenario Enum specifying the task.
        level (int): Level of the task.
        max_steps (int): Maximum number of steps to run.
    """
    env_name = scenario.value
    env = hasard.make(f'{env_name}Level{level}-v0')
    state, info = env.reset()
    terminated = truncated = False
    steps = total_cost = total_reward = 0

    print(f"Running {env_name} Level {level} for up to {max_steps} steps...")

    while steps < max_steps and not (terminated or truncated):
        action = env.action_space.sample()
        state, reward, cost, terminated, truncated, info = env.step(action)
        env.render()
        steps += 1
        total_cost += cost
        total_reward += reward

    env.close()

    print(f"Ran {env_name} Level {level} for {steps} steps.")
    print(f"Reward: {total_reward:.2f}, Cost: {total_cost:.2f}")
    print("Final Statistics:", info.get('env_stats', {}))
    print("=" * 50)

def main():
    max_steps = 250  # Maximum steps per task

    for scenario in hasard.Scenario:
        for level in hasard.LEVELS:
            run_task(scenario, level, max_steps)

if __name__ == '__main__':
    main()
