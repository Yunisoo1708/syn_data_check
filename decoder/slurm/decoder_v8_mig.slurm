#!/bin/bash
#SBATCH --job-name=dec_v8_mig
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v8_%j.log
# Whole-decoder fine-tune run 6 (COORDINATES + CHIRALITY): start = INIT_DEC (default run 5 epoch 0), identity given, pool of 46k peptides + 7,534 mirror-image D-peptides + the 11,291-complex synthetic pool (train_all_cif4, with the new D-form complexes); loss = CE + 0.25 x coordinate error (non-canonical tokens x3) + 0.5 x chirality (signed volume) loss. About 0.13 USD per hour, at most 3 h.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif4 COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export BLIND=0 STRUCT_W=${STRUCT_W:-0.25} NCW=${NCW:-3} CHI_W=${CHI_W:-0.5} INIT=${INIT_DEC:-$R/decoder_v7_498188.pt}
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-64} PBATCH=${PBATCH:-8} EPOCHS=${EPOCHS:-3} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=0 SAVE_PATH=$R/decoder_v8_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v8.py
echo EXIT $?
