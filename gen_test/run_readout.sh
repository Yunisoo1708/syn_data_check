#!/bin/bash
# usage: run_readout.sh WAIT_MARKER_LOG:MARKER TAG name:adapter [name:adapter ...]  -- sequence-readout test at NCAA tokens (eval_readout.py) on real (103) and ham (365 with an NCAA); waits until MARKER appears in LOG
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
LOGF=${1%%:*}; MARK=${1#*:}; TAG=$2; shift 2
while ! grep -q "$MARK" $LOGF 2>/dev/null; do sleep 30; done
for pair in "$@"; do
  n=${pair%%:*}; A=${pair#*:}; [ "$A" != none ] && A=$T/$A
  for set in real ham; do
    if [ $set = ham ]; then export NCLIST=$T/out/eval_dec/ham_nclist.json; else unset NCLIST; fi
    $HOME/rfd4_venv/bin/python eval_readout.py $n $A diag/readout_${n}_$set.json $set 2>&1 | grep -aE "DONE|ERR|Traceback" | tail -5
  done
  echo "readout $n done $(date)"
done
echo READOUT_${TAG}_DONE
