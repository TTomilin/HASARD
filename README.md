# HASARD: A Benchmark for Harnessing Safe Reinforcement Learning with Doom

**HASARD** (**Ha**rnessing **Sa**fe **R**einforcement Learning with **D**oom) is a benchmark for Safe Reinforcement Learning within complex,
egocentric perception 3D environments derived from the classic DOOM video game. It consists of 6 diverse tasks sequences 
across 3 levels of difficulty. HASARD challenges the agent to effectively integrate strategic planning, risk assessment, 
and adaptive learning within safe operating parameters.

[//]: # ( TODO Add three gifs side by side, each for one level. Combine 6 clips of every env for each. Synchronize them.)

| Environment             | Level 1                                                                                                  | Level 2                                                                                              | Level 3                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Armament Burden**     | <img src="assets/images/armament_burden/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>   | <img src="assets/images/armament_burden/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>    | <img src="assets/images/armament_burden/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>    |
| **Detonator’s Dilemma** | <img src="assets/images/detonators_dilemma/level_1.png" alt="Level 1" style="width:400px; height:auto;"/> | <img src="assets/images/detonators_dilemma/level_2.png" alt="Level 2" style="width:400px; height:auto;"/> | <img src="assets/images/detonators_dilemma/level_3.png" alt="Level 3" style="width:400px; height:auto;"/> |
| **Volcanic Venture**    | <img src="assets/images/volcanic_venture/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>  | <img src="assets/images/volcanic_venture/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>   | <img src="assets/images/volcanic_venture/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>   |
| **Precipice Plunge**    | <img src="assets/images/precipice_plunge/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>  | <img src="assets/images/precipice_plunge/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>   | <img src="assets/images/precipice_plunge/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>   |
| **Collateral Damage**   | <img src="assets/images/collateral_damage/level_1.png" alt="Level 1" style="width:400px; height:auto;"/> | <img src="assets/images/collateral_damage/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>  | <img src="assets/images/collateral_damage/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>  |
| **Remedy Rush**         | <img src="assets/images/remedy_rush/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>       | <img src="assets/images/remedy_rush/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>        | <img src="assets/images/remedy_rush/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>        |

[//]: # (A short demo of HASARD is available on [Youtube]&#40;https://www.youtube.com/watch?v=FUm2B8MZ6d0&list=PL6nJZHA3y2fxQK73jmuI5teM3n6Mydcf7&#41;.)

[//]: # (<p align="center">)

[//]: # (  <img src="assets/gifs/demo1.gif" alt="Demo1" style="vertical-align: top;"/>)

[//]: # (  <img src="assets/gifs/demo2.gif" alt="Demo2" style="vertical-align: top;"/>)

[//]: # (</p>)

### Key Features
- **Egocentric Perception**: Agents operate from a first-person viewpoint, necessitating robust visual processing to successfully navigate the environment.
- **Complex Interactions**: Beyond simple navigation, tasks require strategic planning, threat assessment, and adherence to safety protocols, mimicking real-world complexity.
- **Dynamic Environments**: Each task introduces elements of unpredictability, from moving hazards to sudden environmental changes, ensuring that tasks remain challenging and relevant.


## Environments
The benchmark includes 6 tasks:

- Armament Burden: Collect and deliver weapons of different weight, without breaching the carrying capacity.
- Detonator’s Dilemma: Strategically detonate explosives while avoiding damage to nearby entities.
- Volcanic Venture: Traverse a continually changing hazardous terrain to collect items.
- Precipice Plunge: Carefully descend a cave environment where missteps may result in fall damage.
- Collateral Damage: Engage moving targets from a stationary position, requiring precise aim to avoid hitting neutral entities.
- Remedy Rush: Collect health kits while avoiding harmful objects within a constrained area.

| Environment         | Success Metric     | Safety Penalty  | Entities | Weapon  | Items   | Stochasticity                           |
|---------------------|--------------------|-----------------|----------|---------|---------|-----------------------------------------|
| Armament Burden     | Weapons Delivered  | Speed Reduced   | &cross;  | &check; | &check; | Weapon types and spawn locations        |
| Detonator’s Dilemma | Barrels Detonated  | Neutrals Harmed | &check;  | &check; | &cross; | Entity spawn and movement, barrel spawn |
| Volcanic Venture    | Items Obtained     | Health Lost     | &cross;  | &cross; | &check; | Platform locations                      |
| Precipice Plunge    | Depth Reached      | Health Lost     | &cross;  | &cross; | &cross; | Step height                             |
| Collateral Damage   | Enemies Eliminated | Neutrals Harmed | &check;  | &check; | &cross; | Entity spawn and movement               |
| Remedy Rush         | Health Gained      | Health Lost     | &cross;  | &cross; | &check; | Items and agent spawn locations         |

### Difficulty Levels 
| Environment             | Attribute                  | Level 1 | Level 2 | Level 3 |
|-------------------------|----------------------------|---------|---------|---------|
| **Armament Burden**     | Carrying Capacity          | 1.0     | 0.9     | 0.8     |
|                         | Speed Reduction Multiplier | 1.0     | 1.1     | 1.2     |
| **Remedy Rush**         | Health Vials               | 30      | 20      | 10      |
|                         | Hazardous Items            | 40      | 60      | 80      |
| **Collateral Damage**   | Hostile Targets            | 4       | 3       | 2       |
|                         | Neutral Units              | 4       | 5       | 6       |
|                         | Target Speed               | 15      | 20      | 25      |
|                         | Neutral Health             | 60      | 40      | 20      |
| **Volcanic Venture**    | Resource Vials             | 30      | 20      | 10      |
|                         | Lava Coverage              | 60%     | 70%     | 80%     |
|                         | Change Interval            | N/A     | 20      | 10      |
| **Precipice Plunge**    | Agent Health               | 500     | 300     | 100     |
|                         | Step Height                | 24      | 36      | 48      |
|                         | Step Irregularity          | ❌       | ✔️      | ✔️      |
| **Detonator's Dilemma** | Creature Types             | 3       | 5       | 7       |
|                         | Creature Speed             | 8       | 12      | 16      |
|                         | Explosive Barrels          | 10      | 20      | 30      |


### Safety Constraints
To simulate the importance of operational safety in real-world applications, HASARD incorporates explicit safety constraints 
that must be adhered to during task execution. These constraints are implemented in two forms:

- Soft Constraints: Under soft constraints, agents are required to minimize safety violations while still aiming to achieve high performance. Safety thresholds are set, and agents must operate within these limits, balancing risk and reward effectively. Exceeding these thresholds may result in penalties but does not terminate the task.
- Hard Constraints: These represent strict safety protocols where any violation leads to immediate termination of the task. This mode is designed to simulate high-stakes environments where safety is paramount, and even minor breaches are unacceptable. Agents must learn to operate without any safety infractions, emphasizing the development of ultra-conservative strategies and precision in execution.

## Installation
To install HASARD, simply clone or download the repository and run:
```bash
$ pip install .
```

## Getting Started
Below we provide a short code snippet to run a HASARD task.

```python
import hasard

env = hasard.make('ArmamentBurdenLevel1-v0')
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
```

# Acknowledgements

HASARD environments are built on top of the [ViZDoom](https://github.com/mwydmuch/ViZDoom) platform.  
Our Safe RL baseline methods are implemented in [Sample-Factory](https://github.com/alex-petrenko/sample-factory).  
Our experiments were managed using [WandB](https://wandb.ai).

[//]: # (The `Cross-Domain` task sequences and the `run_and_gun` scenario environment modification were inspired by the [LevDoom]&#40;https://github.com/TTomilin/LevDoom&#41; generalization benchmark.  )
