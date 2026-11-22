#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
mkdir -p $T/out/eval
PY=$HOME/rfd4_venv/bin/python
for spec in "control:none" "A_all:$T/out/A_all/best" "B_first:$T/out/B_first/best" "C_long:$T/out/C_long/best"; do
  n=${spec%%:*}; a=${spec#*:}
  [ -f $T/out/eval/$n.json ] && continue
  $PY $T/eval_one.py $n $a $T/out/eval/$n.json 2>&1 | grep -E "^(control|A_all|B_first|C_long)|Traceback|Error" | tail -n 3
done
echo EVAL_ALL_DONE
