#!/bin/bash
#SBATCH --job-name=dec_v9_h200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v9_%j.log
# Whole-decoder fine-tune run 8 (PEPTIDE BONDS) on a full H200: start = INIT_DEC (default run 5b), identity given, same data as run 7 (46k peptides + 7,534 mirror D-peptides + 11,291-complex pool), loss = CE + 0.5 x coordinate error (non-canonical x2) + 0.15 x chirality + BOND_W x flat-bottom C-N bond loss (1.33 +- 0.04 A) on true latents and on noised latents. About 0.9 USD per hour, at most 3 h = at most about 2.7 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif4 COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export BLIND=0 STRUCT_W=${STRUCT_W:-0.5} NCW=${NCW:-2} CHI_W=${CHI_W:-0.15} BOND_W=${BOND_W:-10} BOND_TOL=${BOND_TOL:-0.04} BOND_NOISE=${BOND_NOISE:-0.1} INIT=${INIT_DEC:-$R/decoder_v7b_498314.pt}
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-128} PBATCH=${PBATCH:-16} EPOCHS=${EPOCHS:-4} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=${NUM_WORKERS:-4} SAVE_PATH=$R/decoder_v9_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v9.py
echo EXIT $?
