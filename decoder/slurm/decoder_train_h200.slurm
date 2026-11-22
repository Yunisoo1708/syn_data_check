#!/bin/bash
#SBATCH --job-name=decoder_train
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/decoder_train_%j.log
# Generic decoder fine-tune on one full H200 (about 0.9 USD per hour, at most 3 h = at most about 2.7 USD; run 8 needed 37 min = about 0.6 USD).
# The defaults below are the RUN 8 recipe (the default decoder): peptide-bond loss, no clash loss, start = run 5b. Other runs: override with
#   sbatch --export=ALL,INIT_DEC=<file>,CLASH_W=5,NOISE_SIGMA=0.2 decoder_train_h200.slurm     (run 9)       -- see decoder/PRESETS.md for every run.
# Files needed in $R: decoder_train.py, bond_loss.py, _common.py, rfproteina_encoding_ext.py, the vocabulary yaml, bundle_189, the data folders below.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif4 COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export CLASH_W=${CLASH_W:-0} CLASH_D0=${CLASH_D0:-2.5} NOISE_SIGMA=${NOISE_SIGMA:-0} CA_JIT=${CA_JIT:-0} NOISY_BOND_W=${NOISY_BOND_W:-0} CLIP=${CLIP:-5}
export BLIND=0 STRUCT_W=${STRUCT_W:-0.5} NCW=${NCW:-2} CHI_W=${CHI_W:-0.15} BOND_W=${BOND_W:-10} BOND_TOL=${BOND_TOL:-0.04} BOND_NOISE=${BOND_NOISE:-0} INIT=${INIT_DEC:-$R/decoder_v7b_498314.pt}
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-128} PBATCH=${PBATCH:-16} EPOCHS=${EPOCHS:-4} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=${NUM_WORKERS:-4} SAVE_PATH=$R/decoder_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_train.py
echo EXIT $?
