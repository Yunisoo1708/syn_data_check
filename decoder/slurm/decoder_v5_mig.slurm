#!/bin/bash
#SBATCH --job-name=dec_v5_mig
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=5:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v5_%j.log
# Whole-decoder fine-tune, 189 classes, run 3: the settings of run 2 (job 488524: bundle start, 1,429 synthetic complexes, batch 64/8) + 14,000 mirror-image D-peptides from the TRAIN split of the pool (real D geometry for the 13 new D-forms and the 6 D-alpha codes). Validation and test are exactly run 2's. MIG slice about 0.13 USD per hour.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-14400} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-64} PBATCH=${PBATCH:-8} EPOCHS=${EPOCHS:-4} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=2 SAVE_PATH=$R/decoder_v5_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v5.py
echo EXIT $?
