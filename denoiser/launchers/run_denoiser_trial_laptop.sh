#!/bin/bash
# Denoiser (flow LoRA) trial on the laptop RTX 4060: A = all clean complexes, B = first batch only (dataset-size ablation). Same budget for both.
P=/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch; R=/mnt/c/Research/RFD4_fine_tune
export RFD4_VOCAB_YAML=$P/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="env PYTHONPATH=/home/yunisoo/binder/envs/rfd4_env/lib/python3.12/site-packages:$P:$R /home/yunisoo/binder/envs/rfd3_env/bin/python"
mkdir -p $R/denoiser_trial
echo "A start $(date)"
$PY $P/ncaa_fine_tuning_v2.py /mnt/c/Research/Dataset_ncaa/rfd4_bundle_189 $R/train_all_cif_clean $R/denoiser_trial/A_all --groups $R/complex_groups.json --steps 1000 --bs ${BS:-2} --inner 5 --eval-every 100 --eval-batches 14 --patience 4 --max-seconds 12000 > $R/denoiser_trial/A_all.log 2>&1
echo "A end $(date)"
echo "B start $(date)"
$PY $P/ncaa_fine_tuning_v2.py /mnt/c/Research/Dataset_ncaa/rfd4_bundle_189 $R/train_all_cif_clean $R/denoiser_trial/B_first --groups $R/complex_groups.json --filter _ncf --steps 1000 --bs ${BS:-2} --inner 5 --eval-every 100 --eval-batches 14 --patience 4 --max-seconds 7200 > $R/denoiser_trial/B_first.log 2>&1
echo "B end $(date)"
