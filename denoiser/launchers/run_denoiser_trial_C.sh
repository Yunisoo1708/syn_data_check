#!/bin/bash
# Trial C: longer, steadier denoiser fine-tune on ALL clean complexes (box GPU 0, batch size 1). 3000 steps, 30 held-out batches per eval, patience 8, cap 3 h. Start only after trial B has finished.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out
echo "C start $(date)"
$PY $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/C_long --groups $T/complex_groups.json --steps 3000 --bs 1 --inner 5 --eval-every 150 --eval-batches 30 --patience 8 --lr 1e-4 --max-seconds 10800 > $T/out/C_long_bs1.log 2>&1
echo "C end $(date)"
