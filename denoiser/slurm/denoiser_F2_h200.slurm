#!/bin/bash
#SBATCH --job-name=dn_F2_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_F2_H200_%j.log
# Trial F2: continue trial F from its BEST weights (job 493150, step 300) with a smaller learning rate (5e-5 instead of 2e-4) = "reduce on plateau". Same data, batch 4, one more epoch, early stop after 5 readings without gain, cap 3,600 s of training. About 0.9 USD per hour (1.5 h limit = at most about 1.35 USD).
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v2.py $R/bundle_189 $R/train_all_cif2 $R/dn_F2_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/dn_F_H200_493150/best --epochs 1 --bs 4 --inner 5 --lr ${LR:-5e-5} --eval-every 100 --eval-batches 30 --patience 5 --max-seconds 3600 > $R/dn_F2_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
