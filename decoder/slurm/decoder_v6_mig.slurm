#!/bin/bash
#SBATCH --job-name=dec_v6_mig
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v6_%j.log
# Whole-decoder fine-tune run 4 (BLIND): start = run 3 decoder, same data as run 3 (pool + 1,429 synthetic complexes + 7,534 mirror-image D-peptides), but the residue identity (cond_residue_type) is hidden from the decoder in training and validation, so it must read the residue type from the latent code. About 0.13 USD per hour, at most 3 h.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export BLIND=1 INIT=$R/decoder_v5_493747.pt
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-64} PBATCH=${PBATCH:-8} EPOCHS=${EPOCHS:-4} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=2 SAVE_PATH=$R/decoder_v6_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v6.py
echo EXIT $?
