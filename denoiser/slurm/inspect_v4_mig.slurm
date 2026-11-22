#!/bin/bash
#SBATCH --job-name=dn_v4_insp
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:12:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_v4insp_%j.log
# Inspect the structure of denoiser training batches (which tokens are given context, which chains). MIG slice, at most 12 min = about 0.03 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/inspect_batch_v4.py $R/bundle_189 $R/train_all_cif_bal280 6
echo EXIT $?
