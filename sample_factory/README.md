# Sample Factory Training for HASARD

This directory contains a highly parallelized training implementation for HASARD environments using [Sample Factory](https://github.com/alex-petrenko/sample-factory).

## Quick Start

### Training a Safe RL Agent

To train an agent on a HASARD environment:

```bash
python sample_factory/train.py --algo PPOLag --env volcanic_venture --level 1
```

## Configuration

For detailed information about all available configuration parameters, see the [Configuration Guide](cfg/README.md).

### Key Parameters

- `--env`: Environment name (e.g., 'armament_burden')
- `--algo`: Algorithm (e.g., 'PPO', 'PPOLag', 'PPOCost')
- `--level`: Difficulty level (1, 2, or 3)
- `--num_workers`: Number of parallel worker processes
- `--batch_size`: Batch size for training

## Evaluating Trained Models

After training, evaluate your models using:

```bash
python sample_factory/enjoy.py --algo PPOLag --env armament_burden --level 3 --experiment my_experiment --timestamp
```

### Loading Models for Evaluation

To load a trained model for evaluation, you need to specify:
- `--experiment`: The name of your training experiment
- `--timestamp`: Loads the model from the checkpoint (automatically finds the latest checkpoint)
- `--load_checkpoint_kind`: Choose between 'latest' (default) or 'best' checkpoint

Example with specific checkpoint:
```bash
python sample_factory/enjoy.py --algo PPOLag --env armament_burden --level 3 --experiment my_experiment --timestamp --load_checkpoint_kind best
```

The model configuration and weights will be automatically loaded from the experiment directory.

## Environments

All HASARD environments are available in 3 difficulty levels:
- `Armament Burden`: Navigate while managing weapon weight
- `Detonators Dialemma`: Defuse bombs while avoiding explosions
- `Volcanic Venture`: Cross lava-filled terrain safely
- `Precipice Plunge`: Navigate cliff edges without falling
- `Collateral Damage`: Complete objectives while minimizing civilian casualties
- `Remedy Rush`: Collect medical supplies while avoiding hazards

## Logging and Monitoring

### Weights & Biases Integration

Enable comprehensive logging and monitoring with Weights & Biases:

```bash
python sample_factory/train.py --algo PPOLag --env armament_burden --level 1 --with_wandb --wandb_project my_project --wandb_user my_username
```

### Heatmap Logging

The framework automatically logs agent position heatmaps during training when using Weights & Biases. These heatmaps show:
- Agent movement patterns and exploration behavior
- Evolution of exploration over training time
- Overlay visualizations on environment maps

Heatmaps are logged at regular intervals and can be viewed in the Weights & Biases dashboard under the "Media" section.

### Gameplay Recording

Record gameplay videos during evaluation:

```bash
python sample_factory/enjoy.py --algo PPOLag --env armament_burden --level 3 --experiment my_experiment --timestamp --save_video
```

Key video recording parameters:
- `--save_video`: Enable video recording during evaluation
- `--overwrite_video`: Overwrite existing videos
- `--no_render`: Disable real-time rendering (faster evaluation)

Videos are saved in the experiment directory and automatically uploaded to Weights & Biases when enabled.

## Curriculum Learning

Implement curriculum learning by running multiple experiments, one per level. Train on each level, save the model, and continue with the next level by loading the previous model:

```bash
# Step 1: Train on Level 1
python sample_factory/train.py --algo PPOLag --env armament_burden --level 1 --experiment curriculum_level1

# Step 2: Train on Level 2, loading the Level 1 model
python sample_factory/train.py --algo PPOLag --env armament_burden --level 2 --experiment curriculum_level2 --restart_behavior resume --load_checkpoint_kind best

# Step 3: Train on Level 3, loading the Level 2 model  
python sample_factory/train.py --algo PPOLag --env armament_burden --level 3 --experiment curriculum_level3 --restart_behavior resume --load_checkpoint_kind best
```

Each experiment builds upon the previous level's learned policy, enabling progressive skill development across increasing difficulty levels. The `--restart_behavior resume` flag ensures the model continues training from the previous checkpoint rather than starting from scratch.
