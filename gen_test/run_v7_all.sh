#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
for set in real ham; do MODE=given $HOME/rfd4_venv/bin/python diag_decode_chir.py $T/decoder_v7_COORD.pt diag/v7_given_$set.json $set 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-420; done
echo DIAG7_DONE
for m in control G_ncw10 G2_pool G3_bal; do
  case $m in control) A=none;; *) A=$T/out/$m/best;; esac
  o=$T/gen_test/out/${m}_v7
  $HOME/rfd4_venv/bin/python gen_run.py ${m}_v7 $A $T/decoder_v7_COORD.pt $o "" 8 > $o.log 2>&1
  echo "$m v7 done $(date)"
done
echo V7_ALL_DONE
