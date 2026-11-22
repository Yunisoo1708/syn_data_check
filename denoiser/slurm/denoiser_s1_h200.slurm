#!/bin/bash
#SBATCH --job-name=dn_s1_H200
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_s1_H200_%j.log
# Trial S1 = G3 + a sequence readout trained on NON-CANONICAL identities: the identity of every NCAA token is hidden (--nc-hide-p 1) so the readout (189 classes, attached to the trunk) must infer the NCAA from the structure; weight 3 on the readout CE. Start = G3 best, same balanced data, batch 4, lr 5e-5, 600 steps. A G4-sized run took 19 min = about 0.3 USD; cap 1.5 h = at most 1.35 USD.
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/denoiser_train.py $R/bundle_189 $R/train_all_cif_bal280 $R/dn_s1_H200_$SLURM_JOB_ID --groups $R/complex_groups.json --resume $R/dn_G3_H200_495885/best --steps ${STEPS:-600} --bs 4 --inner 5 --lr ${LR:-5e-5} --eval-every 100 --eval-batches 30 --patience 5 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --eval-wcap 100 --eval-clip 150 --eval-list $R/heldout30_list.txt --train-readout --readout-attach --seq-w ${SEQ_W:-3} --nc-hide-p 1.0 --max-seconds 4800 > $R/dn_s1_H200_$SLURM_JOB_ID.train.log 2>&1
echo EXIT $?
