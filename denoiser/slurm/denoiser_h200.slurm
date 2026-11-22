#!/bin/bash
#SBATCH --job-name=dn_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_H200_%j.log
# Denoiser (flow LoRA) fine-tune on a full H200: ALL clean complexes, 3 epochs, batch 8. Held-out = the same 12% of protein families as the box trials, scored one complex at a time on the first 30 held-out complexes (same as box trial C).
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v2.py $R/bundle_189 $R/train_all_cif $R/dn_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --epochs 3 --bs 8 --inner 5 --lr ${LR:-2e-4} --eval-every 40 --eval-batches 30 --patience 99 --max-seconds 6000 > $R/dn_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
