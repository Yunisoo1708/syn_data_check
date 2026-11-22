#!/bin/bash
# Score trial B's best LoRA (trained on first-batch complexes only) on trial A's held-out complexes (all-data held-out set, same 14 complexes and noise as A's evals).
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
$HOME/rfd4_venv/bin/python $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/eval_B_on_A --groups $T/complex_groups.json --resume $T/out/B_first/best --steps 0 --bs 1 --eval-batches 14 > $T/out/eval_B_on_A.log 2>&1
grep -E "^(complexes|eval step|resumed)|Traceback" $T/out/eval_B_on_A.log | cut -c1-300
