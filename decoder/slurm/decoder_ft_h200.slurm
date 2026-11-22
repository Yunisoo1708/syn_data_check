#!/bin/bash
#SBATCH --job-name=dec_ft_v2
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
B=/gpfs/projects/stf/legia8
cd $B/RFD4_fine_tune
export RFD_BUNDLE=$B/RFD4_fine_tune/models/RFD4_bundle_176 DATA_DIR=$B/NCAA_dataset/training_pool_all BATCH=${BATCH:-1024} EPOCHS=${EPOCHS:-20} NUM_WORKERS=6 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v2.py
echo EXIT $?
