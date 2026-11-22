#!/bin/bash
#SBATCH --job-name=dn_v4_dbgH
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_v4dbgH_%j.log
# Debug of the denoiser clash-loss trainer (ncaa_fine_tuning_v4.py): start = G3 best, 6 steps, batch 2, inner 1, prints flow loss and clash term per step. full H200, at most 20 min = at most 0.3 USD (the MIG slice ran out of memory in the decoder gradient path).
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v4.py $R/bundle_189 $R/train_all_cif_bal280 $R/dn_v4dbgH_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/dn_G3_H200_495885/best --decoder $R/decoder_v9_499602.pt --steps 12 --max-steps-debug 12 --bs 4 --inner 1 --lr 3e-5 --eval-every 1000 --eval-batches 8 --nc-weight 10 --wcap 100 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --clash-w ${CLASH_W:-5} --clash-tmin ${CLASH_TMIN:-0.6} --max-seconds 1200
echo EXIT $?
