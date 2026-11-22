#!/bin/bash
#SBATCH --job-name=dec_v7_mig
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v7_%j.log
# Whole-decoder fine-tune run 5 (COORDINATES): start = run 3 decoder, identity GIVEN as in run 3, same pool + 7,534 mirror-image D-peptides + the 8,947-complex synthetic pool, loss = residue-type CE + 0.25 x atom-coordinate error (A^2), so the atom output is trained on D-forms. About 0.13 USD per hour, at most 3 h.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif3 COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export BLIND=0 STRUCT_W=${STRUCT_W:-0.25} INIT=$R/decoder_v5_493747.pt
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-64} PBATCH=${PBATCH:-8} EPOCHS=${EPOCHS:-3} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=2 SAVE_PATH=$R/decoder_v7_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v7.py
echo EXIT $?
