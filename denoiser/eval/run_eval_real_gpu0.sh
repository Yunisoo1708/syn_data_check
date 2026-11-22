#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out/eval_real
for spec in "control:none" "A_all:$T/out/A_all/best" "B_first:$T/out/B_first/best" "C_long:$T/out/C_long/best" "D_rank64:$T/out/D_rank64/best" "E_rank64_ff:$T/out/E_rank64_ff/best"; do
  n=${spec%%:*}; a=${spec#*:}
  [ -f $T/out/eval_real/$n.json ] && continue
  $PY $T/eval_one_real.py $n $a $T/out/eval_real/$n.json 2>&1 | grep -E "done|Traceback|Error" | tail -n 3
done
echo EVAL_REAL_DONE
