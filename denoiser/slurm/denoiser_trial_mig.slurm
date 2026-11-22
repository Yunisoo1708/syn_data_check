#!/bin/bash
#SBATCH --job-name=dn_trial
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=5:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/dn_trial_%x_%j.log
# Denoiser (flow LoRA) trial on an H200 MIG slice. usage: sbatch --job-name=dn_A --export=ALL,TAG=A_all,FILTER= denoiser_trial_mig.slurm   (B: TAG=B_first,FILTER=_ncf)
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; cd $R
export RFD_BUNDLE=$R/bundle_189 RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export PYTHONPATH=$R:$B/RFD4_fine_tune
PY=$HOME/rfd4_venv/bin/python
for BS in 2 1; do
  echo "== bs $BS $(date)"
  $PY $R/ncaa_fine_tuning_v2.py $R/bundle_189 $R/train_all_cif $R/dn_trial_$TAG --decoder $R/decoder_v4_488524.pt --groups $R/complex_groups.json ${FILTER:+--filter $FILTER} --steps 1000 --bs $BS --inner 5 --eval-every 100 --eval-batches 14 --patience 4 --max-seconds 16000 > $R/dn_trial_${TAG}_bs$BS.log 2>&1
  rc=$?; echo "exit $rc"
  grep -q "out of memory" $R/dn_trial_${TAG}_bs$BS.log && continue
  break
done
echo ALLDONE
