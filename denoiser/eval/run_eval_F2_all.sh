#!/bin/bash
# score trial F2 (H200 continuation, best weights = step 300) on every test: held-out flow loss, real flow loss, non-canonical positions on both sets
T=$HOME/rfd4_trial
until grep -q EVAL_NC_DONE $T/out/eval_nc.log; do sleep 15; done
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
n=F2_H200; A=$T/out/F2_H200/best
[ -f $T/out/eval/$n.json ] || $PY $T/eval_one.py $n $A $T/out/eval/$n.json 2>&1 | grep -E "^$n|Traceback|rror:" | tail -n 2
[ -f $T/out/eval_real/$n.json ] || $PY $T/eval_one_real.py $n $A $T/out/eval_real/$n.json 2>&1 | grep -E "$n|Traceback|rror:" | tail -n 2
for set in real heldout; do [ -f $T/out/eval_nc/${set}_$n.json ] || $PY $T/eval_one_nc.py $n $A $T/out/eval_nc/${set}_$n.json $set 2>&1 | grep -E " done |Traceback|rror:" | tail -n 2; done
echo EVAL_F2_DONE
