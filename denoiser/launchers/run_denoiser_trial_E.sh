#!/bin/bash
# Trial E: rank 64 on attention PLUS the feed-forward layers and the attention gate (the biggest frozen part of the network). Same data/held-out/eval as C and D. Start after trial D.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out
echo "E start $(date)"
$PY $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/E_rank64_ff --groups $T/complex_groups.json --steps 3000 --bs 1 --inner 5 --eval-every 150 --eval-batches 30 --patience 8 --lr 1e-4 --rank 64 --alpha 128 --extra transition.transition.swish_linear,transition.transition.linear_out,mhba.mha.to_g --max-seconds 10800 > $T/out/E_rank64_ff_bs1.log 2>&1
echo "E end $(date)"
