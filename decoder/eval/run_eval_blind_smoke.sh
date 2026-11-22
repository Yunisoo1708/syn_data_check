#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
mkdir -p $T/out/eval_blind
for mode in v1 v2; do MODE=$mode NCLIST=$T/out/eval_dec/ham_nclist.json TS=0.15 $HOME/rfd4_venv/bin/python $T/eval_one_dec_blind.py control none $T/out/eval_blind/smoke_$mode.json heldout $T/decoder_v5_493747.pt 2>&1 | grep -E "Traceback|rror|DONE|done|^1|^2|^3|_NC|^[0-9A-Z]{4}_" | tail -n 8 | cut -c1-400; done
