#!/bin/bash
#SBATCH --job-name=dn_s1_dbg
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:20:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_s1dbg_%j.log
# Debug of trial S1 (NCAA sequence readout): start = G3 best, 8 steps, batch 2, inner 1; prints how many NCAA tokens have their identity given / hidden and the held-out readout accuracy. MIG slice, at most 20 min = about 0.05 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/denoiser_train.py $R/bundle_189 $R/train_all_cif_bal280 $R/dn_s1dbg_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/dn_G3_H200_495885/best --steps 8 --max-steps-debug 8 --bs 2 --inner 1 --lr 5e-5 --eval-every 1000 --eval-batches 6 --nc-weight 10 --wcap 100 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --train-readout --readout-attach --seq-w 3 --nc-hide-p 1.0 --max-seconds 900
echo EXIT $?
