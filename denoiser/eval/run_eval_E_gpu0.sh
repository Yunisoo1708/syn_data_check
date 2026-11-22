#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out/eval $T/out/eval_real
[ -f $T/out/eval/E_rank64_ff.json ] || $PY $T/eval_one.py E_rank64_ff $T/out/E_rank64_ff/best $T/out/eval/E_rank64_ff.json 2>&1 | grep -E "^E_rank64_ff|Traceback|Error" | tail -n 3
[ -f $T/out/eval_real/E_rank64_ff.json ] || $PY $T/eval_one_real.py E_rank64_ff $T/out/E_rank64_ff/best $T/out/eval_real/E_rank64_ff.json 2>&1 | grep -E "^E_rank64_ff|Traceback|Error" | tail -n 3
echo EVAL_E_DONE
