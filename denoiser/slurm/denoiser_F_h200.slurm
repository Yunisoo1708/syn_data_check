#!/bin/bash
#SBATCH --job-name=dn_F_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_F_H200_%j.log
# Trial F on a full H200: denoiser (flow LoRA, rank 16, attention layers) on the 6,457-complex pool (train_all_cif2: first batch + top-ups + balanced general batch), 3 epochs, batch 4 (batch 8 ran out of memory), lr 2e-4, same held-out protocol as the box trials. About 0.9 USD per hour, cap 9,600 s of training (about 2.4 USD).
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v2.py $R/bundle_189 $R/train_all_cif2 $R/dn_F_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --epochs 3 --bs 4 --inner 5 --lr ${LR:-2e-4} --eval-every 100 --eval-batches 30 --patience 8 --max-seconds 9600 > $R/dn_F_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
