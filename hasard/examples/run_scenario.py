import hasard


def main():
    env = hasard.make('ArmamentBurdenLevel1Soft-v0')
    env.reset()
    done = False
    steps = 0
    total_cost = 0
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        state, reward, cost, done, truncated, info = env.step(action)
        env.render()
        steps += 1
        total_cost += cost
        total_reward += reward
    print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}. Cost: {total_cost:.2f}")
    env.close()


if __name__ == '__main__':
    main()
