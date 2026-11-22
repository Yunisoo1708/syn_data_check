#!/bin/bash
# identity-withheld decode tests with the run 3 decoder: usage run_eval_blind.sh "MODES" "SETS" MODEL...   (box GPU 0; MODES = "v1 v2"; SETS = "heldout real ham")
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python; DEC=$T/decoder_v5_493747.pt; MODES=$1; SETS=$2; shift 2; mkdir -p $T/out/eval_blind
for n in "$@"; do
  case $n in control) a=none;; G_ncw10) a=$T/out/G_ncw10/best;; *) a=$T/out/$n/best;; esac
  for mode in $MODES; do for set in $SETS; do
    o=$T/out/eval_blind/${mode}_${set}_$n.json; [ -f $o ] && continue
    if [ $set = ham ]; then MODE=$mode TS=0.1 NCLIST=$T/out/eval_dec/ham_nclist.json $PY $T/eval_one_dec_blind.py $n $a $o $set $DEC 2>&1 | grep -E "Traceback|rror:" | tail -n 2
    else MODE=$mode TS=0.15 $PY $T/eval_one_dec_blind.py $n $a $o $set $DEC 2>&1 | grep -E "Traceback|rror:" | tail -n 2; fi
    echo "$mode $set $n done $(date)"
  done; done
done
echo BLIND_DONE_$$
