#!/bin/bash
#SBATCH --job-name=dec_v3
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v3_%j.log
# Whole-decoder fine-tune, 189 classes. usage: sbatch [--export=ALL,EPOCHS=10,BATCH=1024,...] decoder_v3_h200.slurm
# H200 about 0.9 USD per hour (4 h limit = at most about 3.6 USD); cancel early if the validation loss stops improving.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/pilot_cif PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export BATCH=${BATCH:-1024} PBATCH=${PBATCH:-32} EPOCHS=${EPOCHS:-10} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=6 SAVE_PATH=$R/decoder_v3_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v3.py
echo EXIT $?
