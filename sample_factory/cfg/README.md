# Sample Factory Configuration Guide

This document provides a comprehensive guide to all configuration parameters available in Sample Factory for HASARD environments.

## Overview

Sample Factory uses command-line arguments to configure training and evaluation. All parameters can be specified via command line flags or loaded from configuration files. The configuration system is organized into several categories:

- [Basic CLI Arguments](#basic-cli-arguments)
- [Reinforcement Learning Parameters](#reinforcement-learning-parameters)
- [Model Architecture](#model-architecture)
- [Environment Configuration](#environment-configuration)
- [Evaluation Parameters](#evaluation-parameters)
- [Logging and Monitoring](#logging-and-monitoring)
- [Population-Based Training](#population-based-training)
- [HASARD-Specific Parameters](#hasard-specific-parameters)

## Basic CLI Arguments

### Core Parameters

- `--algo`: Algorithm to use (default: 'APPO')
  - Options: 'APPO', 'PPO', 'PPOLag', 'CPO'
  - APPO is recommended for high-throughput parallel training

- `--env`: Environment name (required)
  - Examples: 'ArmamentBurdenLevel1-v0', 'VolcanicVentureLevel2-v0'

- `--experiment`: Experiment name for organizing results (required)
  - Used for directory naming and logging

- `--train_dir`: Directory to store training results
  - Default: './train_dir'

- `--device`: Device to use for training
  - Options: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
  - Default: 'cuda' if available, otherwise 'cpu'

### Execution Control

- `--serial_mode`: Run in serial mode (single process)
  - Default: False (parallel mode recommended)

- `--async_rl`: Use asynchronous RL updates
  - Default: True (improves throughput)

- `--num_workers`: Number of parallel worker processes
  - Default: Number of CPU cores
  - Recommended: Match your CPU core count

- `--num_envs_per_worker`: Environments per worker process
  - Default: 2
  - Recommended: 8-16 for optimal throughput

## Reinforcement Learning Parameters

### Training Duration

- `--train_for_env_steps`: Total environment steps to train for
  - Default: 10,000,000
  - Recommended: 50,000,000+ for complex environments

- `--train_for_seconds`: Alternative: train for specified seconds
  - Default: None (use env_steps instead)

### Learning Parameters

- `--learning_rate`: Learning rate for the optimizer
  - Default: 0.0001
  - Range: 0.00001 to 0.01

- `--lr_schedule`: Learning rate schedule
  - Options: 'constant', 'kl_adaptive_minibatch', 'kl_adaptive_epoch'
  - Default: 'constant'

- `--gamma`: Discount factor for future rewards
  - Default: 0.99
  - Range: 0.9 to 0.999

- `--reward_scale`: Scale factor for rewards
  - Default: 1.0
  - Useful for normalizing reward magnitudes

### PPO-Specific Parameters

- `--ppo_clip_ratio`: PPO clipping parameter
  - Default: 0.1
  - Range: 0.05 to 0.3

- `--ppo_clip_value`: Value function clipping parameter
  - Default: 1.0
  - Set to 0.0 to disable value clipping

- `--batch_size`: Batch size for training
  - Default: 1024
  - Recommended: 2048-4096 for GPU training

- `--num_batches_per_iteration`: Batches per training iteration
  - Default: 1
  - Increase for more stable gradients

- `--ppo_epochs`: PPO epochs per iteration
  - Default: 1
  - Range: 1-4 (higher values may cause instability)

### Advanced RL Parameters

- `--gae_lambda`: GAE lambda parameter
  - Default: 0.95
  - Range: 0.9 to 0.99

- `--max_grad_norm`: Maximum gradient norm for clipping
  - Default: 4.0
  - Prevents gradient explosion

- `--exploration_loss_coeff`: Coefficient for exploration bonus
  - Default: 0.003
  - Encourages exploration in sparse reward environments

- `--value_loss_coeff`: Coefficient for value function loss
  - Default: 0.5
  - Balance between policy and value learning

## Model Architecture

### Network Architecture

- `--encoder_type`: Type of encoder network
  - Options: 'conv', 'mlp', 'resnet'
  - Default: 'conv' (recommended for visual environments)

- `--encoder_subtype`: Encoder subtype
  - Options: 'convnet_simple', 'convnet_impala'
  - Default: 'convnet_simple'

- `--hidden_size`: Size of hidden layers
  - Default: 512
  - Common values: 256, 512, 1024

- `--encoder_extra_fc_layers`: Extra fully connected layers
  - Default: 1
  - Range: 0-3

### Recurrent Networks

- `--use_rnn`: Use recurrent neural network
  - Default: True
  - Recommended for partially observable environments

- `--rnn_type`: Type of RNN cell
  - Options: 'gru', 'lstm'
  - Default: 'gru' (faster than LSTM)

- `--rnn_num_layers`: Number of RNN layers
  - Default: 1
  - Range: 1-3

### Actor-Critic Architecture

- `--actor_critic_share_weights`: Share weights between actor and critic
  - Default: True
  - False may improve performance but uses more memory

- `--adaptive_stddev`: Use adaptive standard deviation for continuous actions
  - Default: True
  - Helps with exploration in continuous control

## Environment Configuration

### Basic Environment Settings

- `--env_frameskip`: Number of frames to skip
  - Default: 4
  - Higher values increase training speed but reduce control precision

- `--num_envs`: Total number of environments
  - Calculated as: num_workers × num_envs_per_worker
  - Typical range: 64-512

### Observation Processing

- `--obs_subtract_mean`: Subtract mean from observations
  - Default: 0.0
  - Useful for normalizing visual inputs

- `--obs_scale`: Scale factor for observations
  - Default: 255.0
  - Converts uint8 images to [0,1] range

### Action Space

- `--continuous_actions`: Use continuous action space
  - Default: False
  - Set to True for continuous control tasks

- `--normalize_actions`: Normalize continuous actions
  - Default: True
  - Recommended when using continuous actions

## Evaluation Parameters

### Evaluation Control

- `--eval_deterministic`: Use deterministic policy during evaluation
  - Default: True
  - Provides more consistent evaluation results

- `--max_num_episodes`: Maximum episodes for evaluation
  - Default: 1000000
  - Set lower for faster evaluation

- `--max_num_frames`: Maximum frames for evaluation
  - Default: 1000000
  - Alternative stopping criterion

### Video Recording

- `--save_video`: Save evaluation videos
  - Default: False
  - Useful for visualizing policy behavior

- `--video_frames`: Number of frames to record
  - Default: 1000000
  - Set to -1 for unlimited recording

- `--fps`: Target FPS for rendering
  - Default: 0 (no limit)
  - Set to 30-60 for real-time visualization

## Logging and Monitoring

### Weights & Biases Integration

- `--with_wandb`: Enable Weights & Biases logging
  - Default: False
  - Provides comprehensive experiment tracking

- `--wandb_user`: WandB username (entity)
  - Required when using WandB
  - Must be specified from command line

- `--wandb_project`: WandB project name
  - Default: 'sample_factory'
  - Organizes experiments by project

- `--wandb_group`: WandB group name
  - Default: None (uses environment name)
  - Groups related experiments

- `--wandb_tags`: Tags for experiment organization
  - Default: []
  - Helps with filtering and searching experiments

### Local Logging

- `--log_level`: Logging verbosity level
  - Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
  - Default: 'INFO'

- `--save_every_sec`: Save checkpoint every N seconds
  - Default: 120
  - More frequent saves provide better recovery options

- `--keep_checkpoints`: Number of checkpoints to keep
  - Default: 2
  - Prevents disk space issues

### Performance Monitoring

- `--log_heatmap`: Log agent position heatmaps
  - Default: False
  - Useful for analyzing exploration patterns

- `--log_overlay`: Log heatmap overlays on environment maps
  - Default: True
  - Provides visual exploration analysis

- `--heatmap_log_interval`: Interval for heatmap logging
  - Default: 1,000,000 steps
  - Balance between detail and storage

## Population-Based Training

### PBT Configuration

- `--with_pbt`: Enable Population-Based Training
  - Default: False
  - Automated hyperparameter optimization

- `--num_policies`: Number of policies in population
  - Default: 8
  - More policies provide better exploration of hyperparameter space

- `--pbt_period_env_steps`: Steps between PBT updates
  - Default: 5,000,000
  - Frequency of hyperparameter mutations

- `--pbt_start_mutation`: Steps before starting mutations
  - Default: 20,000,000
  - Allows policies to stabilize before mutations

### PBT Hyperparameters

- `--pbt_mix_policies_in_one_env`: Mix policies in single environment
  - Default: False
  - Can improve diversity but may complicate analysis

- `--pbt_target_objective`: Objective to optimize
  - Default: 'true_reward'
  - Options: 'true_reward', 'episode_length', custom metrics

## HASARD-Specific Parameters

### Environment Variants

- `--level`: Difficulty level
  - Options: 1, 2, 3
  - Higher levels increase complexity and challenge

- `--constraint`: Safety constraint type
  - Options: 'soft', 'hard'
  - 'hard' enforces strict safety limits

### Visual Configuration

- `--resolution`: Screen resolution
  - Options: '160x120', '320x240', '640x480', '800x600', '1280x720', '1600x1200'
  - Higher resolutions provide more detail but require more computation

- `--wide_aspect_ratio`: Enable wide aspect ratio
  - Default: False
  - Provides better field of view

- `--resolution_eval`: Resolution for evaluation videos
  - Default: '1280x720'
  - Can be different from training resolution

### Recording and Replay

- `--record`: Enable gameplay recording
  - Default: True
  - Records episodes for analysis

- `--video_dir`: Directory for video storage
  - Default: 'videos'
  - Ensure sufficient disk space

- `--record_every`: Record video every N steps
  - Default: 5000
  - Balance between coverage and storage

- `--video_length`: Length of recorded videos in steps
  - Default: 2100
  - Approximately 1 minute at 35 FPS

## Configuration Best Practices

### Performance Optimization

1. **CPU Utilization**: Set `--num_workers` to match your CPU cores
2. **Memory Management**: Monitor RAM usage; reduce workers if needed
3. **GPU Utilization**: Use larger batch sizes (2048-4096) for better GPU usage
4. **Storage**: Ensure sufficient disk space for checkpoints and videos

### Hyperparameter Tuning

1. **Learning Rate**: Start with 0.0001, adjust based on training stability
2. **Batch Size**: Larger batches provide more stable gradients
3. **Environment Steps**: Train for at least 50M steps for complex environments
4. **Exploration**: Adjust exploration coefficients for sparse reward environments

### Debugging and Monitoring

1. **Logging**: Use INFO level for normal training, DEBUG for troubleshooting
2. **Checkpoints**: Save frequently during long training runs
3. **Videos**: Record periodically to monitor policy behavior
4. **WandB**: Use for comprehensive experiment tracking and comparison

### Safety and Constraints

1. **Constraint Types**: Use 'soft' for learning, 'hard' for deployment
2. **Cost Monitoring**: Track safety violations during training
3. **Evaluation**: Test with deterministic policies for consistent results

## Example Configurations

### Fast Prototyping
```bash
--num_workers=8 --num_envs_per_worker=4 --batch_size=512 --train_for_env_steps=10000000
```

### High Performance Training
```bash
--num_workers=72 --num_envs_per_worker=16 --batch_size=2048 --train_for_env_steps=100000000
```

### Population-Based Training
```bash
--with_pbt=True --num_policies=8 --pbt_period_env_steps=5000000 --pbt_start_mutation=20000000
```

### Monitoring
```bash
--with_wandb=True --save_video=True --record_every=10000
```

For more detailed information about specific parameters, refer to the source code in `cfg.py` or use the `--help` flag with the training scripts.