#!/bin/bash
# Trial D: rank 16 -> 64 on the SAME layers as trial C (attention to_qkv and to_out_node, 14 blocks), alpha 128 (alpha = 2 x rank as before). Same data, held-out set (first 30 complexes), noise and eval as trial C, so the two are directly comparable.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out
echo "D start $(date)"
$PY $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/D_rank64 --groups $T/complex_groups.json --steps 3000 --bs 1 --inner 5 --eval-every 150 --eval-batches 30 --patience 8 --lr 1e-4 --rank 64 --alpha 128 --max-seconds 10800 > $T/out/D_rank64_bs1.log 2>&1
echo "D end $(date)"
