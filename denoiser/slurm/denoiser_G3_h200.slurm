#!/bin/bash
#SBATCH --job-name=dn_G3_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_G3_H200_%j.log
# Trial G3: same as G2 (start = trial G best weights, G recipe, batch 4, lr 5e-5, same clipped held-out check on the same 30 complexes) but trained on the CLASS-BALANCED subset of the pool: cap 280 positions per swapped code = 2,232 training complexes (every one of the 25 codes has 273 to 280 positions) + the 1,021 held-out complexes of the 9,113-complex pool. Only the data differs from G2. At most 2 h = at most about 1.8 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v3.py $R/bundle_189 $R/train_all_cif_bal280 $R/dn_G3_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/G_ncw10_best --steps 3000 --bs 4 --inner 5 --lr ${LR:-5e-5} --eval-every 100 --eval-batches 30 --patience 5 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --max-seconds 6000 > $R/dn_G3_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
