# Sample Factory Training for HASARD

This directory contains a highly parallelized training implementation for HASARD environments using [Sample Factory](https://github.com/alex-petrenko/sample-factory).

## Features

- **Safe RL Support**: Built-in support for cost constraints
- **Comprehensive Logging**: Integration with Weights & Biases (WandB) for experiment tracking
- **Video Recording**: Automatic gameplay recording for policy visualization

## Quick Start

### Training a Safe RL Agent

To train an agent on a HASARD environment:

```bash
python sample_factory/train.py --env=ArmamentBurdenLevel1-v0 --experiment=my_experiment
```

### Basic Training Options

```bash
# Train on different environments and difficulty levels
python sample_factory/train.py --env=VolcanicVentureLevel2-v0 --level=2 --experiment=volcanic_l2

# Use hard safety constraints instead of soft
python sample_factory/train.py --env=DetonatorsDialemmaLevel1-v0 --constraint=hard --experiment=dd_hard

# Adjust training parameters
python sample_factory/train.py --env=PrecipicePlungeLevel3-v0 --num_workers=16 --num_envs_per_worker=8 --experiment=pp_parallel
```

### Advanced Training Configuration

```bash
# High-performance parallel training with multiple GPUs
python sample_factory/train.py \
    --env=CollateralDamageLevel2-v0 \
    --experiment=cd_advanced \
    --num_workers=32 \
    --num_envs_per_worker=16 \
    --batch_size=2048 \
    --learning_rate=0.0001 \
    --train_for_env_steps=50000000

# Population-Based Training for hyperparameter optimization
python sample_factory/train.py \
    --env=RemedyRushLevel1-v0 \
    --experiment=rr_pbt \
    --with_pbt=True \
    --pbt_mix_policies_in_one_env=False \
    --pbt_period_env_steps=5000000 \
    --num_policies=8
```

## Environment-Specific Parameters

### HASARD-Specific Options

- `--level`: Difficulty level (1, 2, or 3)
- `--constraint`: Safety constraint type ('soft' or 'hard')
- `--resolution`: Screen resolution ('160x120', '320x240', '640x480', '800x600', '1280x720', '1600x1200')
- `--wide_aspect_ratio`: Enable wide aspect ratio for better field of view

### Rendering and Recording

- `--record`: Enable/disable gameplay recording (default: True)
- `--video_dir`: Directory to save recorded videos (default: 'videos')
- `--record_every`: Record video every N steps (default: 5000)
- `--video_length`: Length of recorded videos in steps (default: 2100)
- `--resolution_eval`: Resolution for evaluation videos (default: '1280x720')

## Evaluating Trained Models

After training, evaluate your models using:

```bash
python sample_factory/enjoy.py --env=ArmamentBurdenLevel1-v0 --experiment=my_experiment
```

### Evaluation Options

```bash
# Evaluate with video recording
python sample_factory/enjoy.py \
    --env=VolcanicVentureLevel2-v0 \
    --experiment=volcanic_l2 \
    --save_video \
    --video_dir=evaluation_videos

# Evaluate multiple episodes
python sample_factory/enjoy.py \
    --env=DetonatorsDialemmaLevel1-v0 \
    --experiment=dd_hard \
    --num_episodes=100
```

## Available HASARD Environments

All HASARD environments are available with three difficulty levels:

### Scenarios
- `ArmamentBurdenLevel{1,2,3}-v0`: Navigate while managing weapon weight
- `DetonatorsDialemmaLevel{1,2,3}-v0`: Defuse bombs while avoiding explosions
- `VolcanicVentureLevel{1,2,3}-v0`: Cross lava-filled terrain safely
- `PrecipicePlungeLevel{1,2,3}-v0`: Navigate cliff edges without falling
- `CollateralDamageLevel{1,2,3}-v0`: Complete objectives while minimizing civilian casualties
- `RemedyRushLevel{1,2,3}-v0`: Collect medical supplies while avoiding hazards

## Performance Optimization

### Parallel Training Tips

1. **CPU Cores**: Set `--num_workers` to match your CPU core count
2. **Environments per Worker**: Use `--num_envs_per_worker=8-16` for optimal throughput
3. **Batch Size**: Increase `--batch_size` for better GPU utilization (try 1024, 2048, 4096)
4. **Memory**: Monitor RAM usage; reduce workers if running out of memory

### GPU Optimization

```bash
# Multi-GPU training
python sample_factory/train.py \
    --env=ArmamentBurdenLevel3-v0 \
    --experiment=ab_multigpu \
    --num_workers=64 \
    --num_envs_per_worker=8 \
    --batch_size=4096 \
    --device=cuda:0
```

## Hyperparameter Recommendations

### For Fast Prototyping
```bash
--num_workers=8 --num_envs_per_worker=4 --batch_size=512 --train_for_env_steps=10000000
```

### For High Performance
```bash
--num_workers=32 --num_envs_per_worker=16 --batch_size=2048 --train_for_env_steps=100000000
```

### For Population-Based Training
```bash
--with_pbt=True --num_policies=8 --pbt_period_env_steps=5000000 --pbt_start_mutation=20000000
```

## Logging and Monitoring

### Weights & Biases Integration

Enable WandB logging for comprehensive experiment tracking:

```bash
python sample_factory/train.py \
    --env=ArmamentBurdenLevel1-v0 \
    --experiment=my_experiment \
    --with_wandb=True \
    --wandb_project=hasard_training \
    --wandb_group=armament_burden
```

### Local Logging

Training logs and checkpoints are saved to:
- Logs: `sample_factory/train_dir/{experiment_name}/`
- Models: `sample_factory/train_dir/{experiment_name}/checkpoint_*.pth`
- Videos: `{video_dir}/{experiment_name}/`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--num_workers` or `--num_envs_per_worker`
2. **Slow Training**: Increase `--batch_size` and ensure GPU utilization
3. **Environment Errors**: Check that HASARD is properly installed (`pip install HASARD`)
4. **Video Recording Issues**: Ensure sufficient disk space in `--video_dir`

### Performance Monitoring

Monitor training progress with:
```bash
# View training logs
tail -f sample_factory/train_dir/{experiment_name}/sf_log.txt

# Check GPU utilization
nvidia-smi

# Monitor system resources
htop
```

## Advanced Features

### Custom Model Architectures

The framework supports custom neural network architectures optimized for HASARD's visual complexity. Models are automatically configured for the specific observation spaces of each environment.

### Curriculum Learning

Implement curriculum learning by progressively increasing difficulty:

```bash
# Start with Level 1, then progress to higher levels
python sample_factory/train.py --env=ArmamentBurdenLevel1-v0 --experiment=curriculum_step1
python sample_factory/train.py --env=ArmamentBurdenLevel2-v0 --experiment=curriculum_step2 --restart_behavior=resume
```

### Safety Constraint Tuning

Experiment with different safety constraint configurations:

```bash
# Soft constraints with different cost budgets
python sample_factory/train.py --env=VolcanicVentureLevel2-v0 --constraint=soft --experiment=soft_constraints

# Hard constraints for strict safety requirements
python sample_factory/train.py --env=VolcanicVentureLevel2-v0 --constraint=hard --experiment=hard_constraints
```

## Contributing

When adding new features or environments:
1. Follow the existing code structure in `sample_factory/doom/`
2. Add appropriate configuration options in `doom_params.py`
3. Test with both serial and parallel training modes
4. Update this README with new features

## Support

For issues specific to the Sample Factory implementation:
- Check the [Sample Factory documentation](https://github.com/alex-petrenko/sample-factory)
- Review training logs in `sample_factory/train_dir/{experiment_name}/`
- Monitor system resources during training

For HASARD environment issues, refer to the main project README.
