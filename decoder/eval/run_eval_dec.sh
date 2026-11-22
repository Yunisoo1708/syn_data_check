#!/bin/bash
# denoise-then-decode scoring with the SAME retrained decoder for every model; usage: run_eval_dec.sh MODEL...   (box GPU 0)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python; DEC=$T/decoder_v4_488524.pt
mkdir -p $T/out/eval_dec
python3 - <<PY
import json
d=json.load(open("$T/out/eval_nc/ham_control.json"))["results"]
json.dump([f for f,v in d.items() if "codes" in v and any(not c.startswith("_") for c in v["codes"])], open("$T/out/eval_dec/ham_nclist.json","w"))
PY
for n in "$@"; do
  case $n in control) a=none;; A_all|B_first|C_long|D_rank64|E_rank64_ff|F_pool2) a=$T/out/$n/best;; F_H200) a=$T/out/F_H200/best;; F2_H200) a=$T/out/F2_H200/best;; esac
  for set in real heldout ham; do
    o=$T/out/eval_dec/${set}_$n.json; [ -f $o ] && continue
    if [ $set = ham ]; then TS=0.1 NCLIST=$T/out/eval_dec/ham_nclist.json $PY $T/eval_one_dec.py $n $a $o $set $DEC 2>&1 | grep -E " done |Traceback|rror:" | tail -n 2
    else TS=0.05,0.15,0.3 $PY $T/eval_one_dec.py $n $a $o $set $DEC 2>&1 | grep -E " done |Traceback|rror:" | tail -n 2; fi
  done
done
echo EVAL_DEC_DONE_$$
