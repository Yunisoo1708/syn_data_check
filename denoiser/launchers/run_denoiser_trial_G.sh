#!/bin/bash
# Trial G: C recipe (rank 16 attention, 1,266 training complexes, batch 1, lr 1e-4) + the three v3 changes: 10x loss weight on non-canonical positions, noise-level weight capped at 100, learning rate halved (twice) after 3 held-out checks without a new best. Box GPU 0.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
echo "G start $(date)"
$HOME/rfd4_venv/bin/python $T/patch/ncaa_fine_tuning_v3.py $T/bundle_189 $T/train_all_cif_clean $T/out/G_ncw10 --groups $T/complex_groups.json --steps 3000 --bs 1 --inner 5 --eval-every 150 --eval-batches 30 --patience 3 --lr 1e-4 --nc-weight 10 --wcap 100 --plateau 0.5 --plateau-evals 3 --max-decays 2 --max-seconds 10800 > $T/out/G_ncw10_bs1.log 2>&1
echo "G end $(date)"
