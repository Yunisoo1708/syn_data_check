#!/bin/bash
#SBATCH --job-name=dec_v10_mig
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:25:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dec_v10dbg_%j.log
# Whole-decoder fine-tune run 9 (CLASH) on a full H200: start = INIT_DEC (default run 8), loss = run 8 loss + CLASH_W x peptide-receptor clash loss (true latents and perturbed latents). About 0.9 USD per hour, at most 3 h = at most about 2.7 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189
cd $R
export RFD_BUNDLE=$R/bundle_189 POOL_DIR=$B/NCAA_dataset/training_pool_189 PILOT_DIR=$R/train_all_cif4 COMPLEX_GROUPS=$R/complex_groups.json PATCH_DIR=$R RFD4_CODE_DIR=$B/RFD4_fine_tune
export CLASH_W=${CLASH_W:-5} CLASH_D0=${CLASH_D0:-2.5} NOISE_SIGMA=${NOISE_SIGMA:-0.15} CA_JIT=${CA_JIT:-0} NOISY_BOND_W=${NOISY_BOND_W:-2} CLIP=${CLIP:-5}
export BLIND=0 STRUCT_W=${STRUCT_W:-0.5} NCW=${NCW:-2} CHI_W=${CHI_W:-0.15} BOND_W=${BOND_W:-10} BOND_TOL=${BOND_TOL:-0.04} BOND_NOISE=${BOND_NOISE:-0} INIT=${INIT_DEC:-$R/decoder_v9_499602.pt}
export MIRROR_LIST=$R/pool_mirror3/mirror_list.txt
export PATIENCE=${PATIENCE:-3} MAX_SECONDS=${MAX_SECONDS:-9000} VAL_PCT=${VAL_PCT:-12}
export BATCH=${BATCH:-64} PBATCH=${PBATCH:-8} EPOCHS=${EPOCHS:-4} LAMBDA=${LAMBDA:-1.0} NUM_WORKERS=${NUM_WORKERS:-0} SAVE_PATH=$R/decoder_v10dbg_${SLURM_JOB_ID}.pt
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
$HOME/rfd4_venv/bin/python decoder_fine_tune_v10.py
echo EXIT $?
