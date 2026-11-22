#!/bin/bash
#SBATCH --job-name=dec_smoke
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:30:00
B=/gpfs/projects/stf/legia8
cd $B/RFD4_fine_tune
export RFD_BUNDLE=$B/RFD4_fine_tune/models/RFD4_bundle_176 DATA_DIR=$B/NCAA_dataset/training_pool_all N_FILES=400 BATCH=64 EPOCHS=1 NUM_WORKERS=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v2.py
echo EXIT $?
