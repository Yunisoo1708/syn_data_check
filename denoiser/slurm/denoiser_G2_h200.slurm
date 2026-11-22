#!/bin/bash
#SBATCH --job-name=dn_G2_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:20:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_G2_H200_%j.log
# Trial G2: continue trial G (best weights = step 2,700 of the box run) on the 8,947-complex pool (train 7,938 / held-out 1,009, first 30 held-out = the benchmark list, same family hash split, 0 families shared with the 163-complex benchmark), v3 recipe: 10x loss weight on non-canonical positions, noise weight capped at 100, learning rate halved on plateau (x2). Batch 4, lr 5e-5, up to 3,000 steps (about 1.7 epochs), training cap 4 h. About 0.9 USD per hour = at most about 3.9 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v3.py $R/bundle_189 $R/train_all_cif3 $R/dn_G2_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/G_ncw10_best --steps 3000 --bs 4 --inner 5 --lr ${LR:-5e-5} --eval-every 100 --eval-batches 30 --patience 5 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --max-seconds 14400 > $R/dn_G2_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
