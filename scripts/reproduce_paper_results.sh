#!/bin/bash

# Define parameter values
methods=("PPO" "PPOPID" "PPOSaute" "PPOLag" "PPOCost" "P3O")
envs=("armament_burden" "volcanic_venture" "remedy_rush" "collateral_damage" "precipice_plunge" "detonators_dilemma")
levels=(1 2 3)
seeds=(1 2 3 4 5)

# Loop over parameter combinations and submit Slurm jobs
for algo in "${methods[@]}"; do
    for env in "${envs[@]}"; do
        for level in "${levels[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "Submitting job for combination: Algo=$algo, Env=$env, Level=$level, Seed=$seed"

                # Create an SBATCH script
                cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --time 05:00:00
#SBATCH --gres gpu:1
#SBATCH -o ~/slurm/%j_"${algo}"_"${env}"_Level_${level}_$(date +%Y-%m-%d-%H-%M-%S).out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hasard
if [ \$? -eq 0 ]; then
    python3 ~/hasard/sample_factory/doom/train_vizdoom.py \
        --algo "$algo" \
        --env "$env" \
        --level $level \
        --seed $seed \
        --train_for_env_steps 5e8 \
	      --with_wandb True \
        --wandb_project hasard
else
    echo "Failed to activate conda environment."
fi
conda deactivate
EOF
            done
        done
    done
done
