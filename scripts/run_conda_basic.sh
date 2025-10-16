#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hasard
python ~/hasard/sample_factory/doom/train_vizdoom.py
conda deactivate