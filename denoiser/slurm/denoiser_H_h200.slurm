#!/bin/bash
#SBATCH --job-name=dn_H_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:40:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_H_H200_%j.log
# Trial H: G's recipe (start = trial G best weights, 10x non-canonical weight, noise weight cap 100, lr halving on plateau, clipped held-out check on the same 30 complexes) on the class-balanced subset of the 11,291-complex pool (cap 600 per swapped code = 5,021 training complexes) PLUS 3,500 MIRROR-IMAGE complexes (whole complexes mirrored: D-protein with mostly D-peptide, real D-geometry; train split only) + the pool's 1,298 held-out complexes. Batch 4, lr 5e-5. At most 2.5 h = at most about 2.3 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v3.py $R/bundle_189 $R/train_all_cif_H $R/dn_H_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/G_ncw10_best --steps 3000 --bs 4 --inner 5 --lr ${LR:-5e-5} --eval-every 100 --eval-batches 30 --patience 5 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --max-seconds 8400 > $R/dn_H_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
