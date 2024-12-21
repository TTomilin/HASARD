import hasard


def main():
    env = hasard.make('ArmamentBurdenLevel1-v0')
    env.reset()
    terminated = truncated = False
    steps = total_cost = total_reward = 0
    while not terminated or not truncated:
        action = env.action_space.sample()
        state, reward, cost, terminated, truncated, info = env.step(action)
        env.render()
        steps += 1
        total_cost += cost
        total_reward += reward
    print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}. Cost: {total_cost:.2f}")
    print("Final Statistics:", info.get('env_stats', {}))
    env.close()


if __name__ == '__main__':
    main()
