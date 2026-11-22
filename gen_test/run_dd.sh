#!/bin/bash
# denoise-then-decode all-atom test per NCAA position (eval_dd_atoms.py): decoder run 8, denoisers control / G / G2 / G3, sets real (103) and ham (365 with an NCAA), t = 0.05, 0.15, 0.3. Waits for the free-sequence runs to finish (one GPU).
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
while ! grep -q FREE_ALL_DONE run_free.log 2>/dev/null; do sleep 30; done
for dn in control G_ncw10 G2_pool G3_bal; do
  case $dn in control) A=none;; *) A=$T/out/$dn/best;; esac
  for set in real ham; do
    if [ $set = ham ]; then export NCLIST=$T/out/eval_dec/ham_nclist.json; else unset NCLIST; fi
    $HOME/rfd4_venv/bin/python eval_dd_atoms.py $dn $A $T/decoder_v10_BOND.pt diag/dd_${dn}_$set.json $set 2>&1 | grep -E "DONE|ERR|Traceback" | tail -4
  done
  echo "dd $dn done $(date)"
done
echo DD_ALL_DONE
