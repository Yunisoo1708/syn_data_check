#!/bin/bash
#SBATCH --job-name=dec_ft_v2b
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
B=/gpfs/projects/stf/legia8
cd $B/RFD4_fine_tune
export RFD_BUNDLE=$B/RFD4_fine_tune/models/RFD4_bundle_176 DATA_DIR=$B/NCAA_dataset/training_pool_all BATCH=${BATCH:-1024} EPOCHS=${EPOCHS:-70} NUM_WORKERS=6 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RESUME=$B/RFD4_fine_tune/weights/decoder/best_decoder_run1.pt SAVE_PATH=$B/RFD4_fine_tune/weights/decoder/best_decoder_run2.pt
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v2.py
echo EXIT $?
