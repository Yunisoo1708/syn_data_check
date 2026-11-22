#!/bin/bash
# score trial F (box run, best weights) on the 163 held-out complexes and the 103 real complexes, same per-complex seeds as every other model
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out/eval $T/out/eval_real
n=F_pool2
[ -f $T/out/eval/$n.json ] || $PY $T/eval_one.py $n $T/out/F_pool2/best $T/out/eval/$n.json 2>&1 | grep -E "^$n|Traceback|Error" | tail -n 3
[ -f $T/out/eval_real/$n.json ] || $PY $T/eval_one_real.py $n $T/out/F_pool2/best $T/out/eval_real/$n.json 2>&1 | grep -E "$n|Traceback|Error" | tail -n 3
echo EVAL_F_DONE
