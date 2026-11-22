#!/bin/bash
# Run the untuned held-out evaluation twice (10 complexes) in separate processes and compare: identical numbers = deterministic.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
for i in 1 2; do
  $HOME/rfd4_venv/bin/python $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/det_test_$i --groups $T/complex_groups.json --steps 0 --bs 1 --eval-batches 10 2>&1 | grep -E "^eval step 0" | cut -c1-90
done
