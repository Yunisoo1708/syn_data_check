#!/bin/bash
# scores G3 (G on the balanced subset, best = step 1,100) on every test (box GPU 0): held-out flow loss, real 103, non-canonical sets, denoise-then-decode with the run 3 decoder
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python; A=$T/out/G3_bal/best; n=G3_bal
mkdir -p $T/out/eval $T/out/eval_real $T/out/eval_nc $T/out/eval_dec5
[ -f $T/out/eval/$n.json ] || $PY $T/eval_one.py $n $A $T/out/eval/$n.json 2>&1 | grep -E "Traceback|rror:" | tail -n 2
echo "held-out done $(date)"
[ -f $T/out/eval_real/$n.json ] || $PY $T/eval_one_real.py $n $A $T/out/eval_real/$n.json 2>&1 | grep -E "Traceback|rror:" | tail -n 2
echo "real done $(date)"
for set in real heldout ham; do [ -f $T/out/eval_nc/${set}_$n.json ] || $PY $T/eval_one_nc.py $n $A $T/out/eval_nc/${set}_$n.json $set 2>&1 | grep -E "Traceback|rror:" | tail -n 2; echo "nc $set done $(date)"; done
bash $T/run_eval_dec5.sh G3_bal
echo EVAL_G2_DONE
