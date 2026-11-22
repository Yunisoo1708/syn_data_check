#!/bin/bash
#SBATCH --job-name=dn_v4_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_v4_H200_%j.log
# Denoiser trial G4 = G3 + clash loss through the frozen run 8 decoder (ncaa_fine_tuning_v4.py). Start = G3 best, same data (class-balanced bal280), batch 4, inner 5, lr 3e-5, held-out check = same 30 complexes plus the weighted clash term. At most 2 h = at most about 1.8 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/ncaa_fine_tuning_v4.py $R/bundle_189 $R/train_all_cif_bal280 $R/dn_v4_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/dn_G3_H200_495885/best --decoder $R/decoder_v9_499602.pt --steps ${STEPS:-600} --bs 4 --inner 5 --lr ${LR:-3e-5} --eval-every 100 --eval-batches 30 --patience 5 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --clash-w ${CLASH_W:-5} --clash-tmin ${CLASH_TMIN:-0.6} --max-seconds 6000 > $R/dn_v4_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
