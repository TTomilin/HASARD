# HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents

**HASARD** (**Ha**rnessing **Sa**fe **R**einforcement Learning with **D**oom) is a benchmark for Safe Reinforcement 
Learning within complex, egocentric perception 3D environments derived from the classic DOOM video game. It features 6 
diverse scenarios each spanning across 3 levels of difficulty.

## 🔗 Useful Links
- 🌐 [Project Page](https://sites.google.com/view/hasard-bench/)
- 🎥 [Short Presentation](https://www.youtube.com/watch?v=A-uKxVVKfvo)
- 🎮 [Demo Video](https://www.youtube.com/watch?v=A-uKxVVKfvo)


<p align="center">
  <img src="assets/gifs/HASARD_Short_1.gif" alt="Demo1" width="49%"/>
  <img src="assets/gifs/HASARD_Short_2.gif" alt="Demo2" width="49%"/>
</p>


| Scenario                | Level 1                                                                                                  | Level 2                                                                                              | Level 3                                                                                              |
|-------------------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Armament Burden**     | <img src="assets/images/armament_burden/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>   | <img src="assets/images/armament_burden/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>    | <img src="assets/images/armament_burden/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>    |
| **Detonator’s Dilemma** | <img src="assets/images/detonators_dilemma/level_1.png" alt="Level 1" style="width:400px; height:auto;"/> | <img src="assets/images/detonators_dilemma/level_2.png" alt="Level 2" style="width:400px; height:auto;"/> | <img src="assets/images/detonators_dilemma/level_3.png" alt="Level 3" style="width:400px; height:auto;"/> |
| **Volcanic Venture**    | <img src="assets/images/volcanic_venture/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>  | <img src="assets/images/volcanic_venture/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>   | <img src="assets/images/volcanic_venture/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>   |
| **Precipice Plunge**    | <img src="assets/images/precipice_plunge/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>  | <img src="assets/images/precipice_plunge/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>   | <img src="assets/images/precipice_plunge/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>   |
| **Collateral Damage**   | <img src="assets/images/collateral_damage/level_1.png" alt="Level 1" style="width:400px; height:auto;"/> | <img src="assets/images/collateral_damage/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>  | <img src="assets/images/collateral_damage/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>  |
| **Remedy Rush**         | <img src="assets/images/remedy_rush/level_1.png" alt="Level 1" style="width:400px; height:auto;"/>       | <img src="assets/images/remedy_rush/level_2.png" alt="Level 2" style="width:400px; height:auto;"/>        | <img src="assets/images/remedy_rush/level_3.png" alt="Level 3" style="width:400px; height:auto;"/>        |


### Key Features
- **Egocentric Perception**: Agents learn solely from first-person pixel observations under partial observability.
- **Beyond Simple Navigation**: Whereas prior benchmarks merely require the agent to reach goal locations on flat surfaces while avoiding obstacles, HASARD necessitates comprehending complex environment dynamics, anticipating the movement of entities, and grasping spatial relationships. 
- **Dynamic Environments**: HASARD features random spawns, unpredictably moving units, and terrain that is constantly moving or periodically changing.
- **Difficulty Levels**: Higher levels go beyond parameter adjustments, introducing entirely new elements and mechanics.
- **Reward-Cost Trade-offs**: Rewards and costs are closely intertwined, with tightening cost budget necessitating a sacrifice of rewards.
- **Safety Constraints**: Each scenario features a hard constraint setting, where any error results in immediate in-game penalties.
- **Focus on Safety**: Achieving high rewards is straightforward, but doing so while staying within the safety budget demands learning complex and nuanced behaviors. 


### Policy Visualization
HASARD enables overlaying a heatmap of the agent's most frequently visited locations providing further insights into its policy and behavior within the environment.
These examples show how an agent navigates Volcanic Venture, Remedy Rush, and Armament Burden during the course of training:

<p align="center">
  <img src="assets/gifs/PPO_volcanic_venture.gif" alt="Volcanic Venture" width="32%">
  <img src="assets/gifs/PPOPID_remedy_rush.gif" alt="Remedy Rush" width="25.7%">
  <img src="assets/gifs/PPOPID_armament_burden.gif" alt="Armament Burden" width="32%">
</p>

### Augmented Observations
HASARD supports augmented observation modes for further visual analysis. By utilizing privileged game state information, 
it can generate simplified observation representations, such as segmenting objects in the scene or rendering the 
environment displaying only depth from surroundings.

<p align="center">
  <img src="assets/gifs/AB_L2_PPOPID_Segment.gif" alt="Demo1" width="49%"/>
  <img src="assets/gifs/DD_L1_PPOPID_depth.gif" alt="Demo2" width="49%"/>
</p>

## Installation
HASARD supports modular installation to install only the dependencies you need:

```bash
# Core dependencies only (environments and basic functionality)
pip install HASARD

# With sample-factory support for training RL agents
pip install HASARD[sample-factory]

# With results analysis and plotting tools
pip install HASARD[results]

# Full installation with all optional dependencies
pip install HASARD[sample-factory,results]
```

To install from source:
```bash
git clone https://github.com/TTomilin/HASARD
cd HASARD
pip install .  # or pip install .[sample-factory,results] for extras
```

## Getting Started
To get started with HASARD, here's a minimal example of running a task environment.
This script can also be found in [`run_env.py`](hasard/examples/run_env.py):

```python
import hasard

env = hasard.make('RemedyRushLevel1-v0')
env.reset()
terminated = truncated = False
steps = total_cost = total_reward = 0
while not (terminated or truncated):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    steps += 1
    total_cost += info['cost']
    total_reward += reward
print(f"Episode finished in {steps} steps. Reward: {total_reward:.2f}. Cost: {total_cost:.2f}")
env.close()
```

## Training
For highly parallelized training of Safe RL agents on HASARD environments, and to reproduce the results from the paper, 
refer to [`sample_factory`](sample_factory/) for detailed usage instructions and examples.

# Acknowledgements
HASARD environments are built on top of the [ViZDoom](https://github.com/mwydmuch/ViZDoom) platform.  
Our Safe RL baseline methods are implemented in [Sample-Factory](https://github.com/alex-petrenko/sample-factory).  
Our experiments were managed using [WandB](https://wandb.ai).

# Citation
If you use our work in your research, please cite it as follows:
```
@inproceedings{tomilin2025hasard,
    title={HASARD: A Benchmark for Vision-Based Safe Reinforcement Learning in Embodied Agents},
    author={T. Tomilin, M. Fang, and M. Pechenizkiy},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```
