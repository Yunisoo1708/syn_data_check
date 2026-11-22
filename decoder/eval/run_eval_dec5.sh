#!/bin/bash
# denoise-then-decode with the RUN 3 decoder (decoder_v5_493747.pt, mirror-image D-peptides) for the given models; outputs out/eval_dec5/{set}_{model}.json; usage: run_eval_dec5.sh MODEL...   (box GPU 0)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python; DEC=$T/decoder_v5_493747.pt; mkdir -p $T/out/eval_dec5
for n in "$@"; do
  case $n in control) a=none;; G_ncw10) a=$T/out/G_ncw10/best;; *) a=$T/out/$n/best;; esac
  for set in real heldout ham; do
    o=$T/out/eval_dec5/${set}_$n.json; [ -f $o ] && continue
    if [ $set = ham ]; then TS=0.1 NCLIST=$T/out/eval_dec/ham_nclist.json $PY $T/eval_one_dec.py $n $a $o $set $DEC 2>&1 | grep -E "Traceback|rror:" | tail -n 2
    else TS=0.05,0.15,0.3 $PY $T/eval_one_dec.py $n $a $o $set $DEC 2>&1 | grep -E "Traceback|rror:" | tail -n 2; fi
  done
done
echo EVAL_DEC5_DONE_$$
