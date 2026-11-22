#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
for cfg in "v5 given" "v5 hidden" "v6 hidden"; do set -- $cfg
  case $1 in v5) D=$T/decoder_v5_493747.pt;; v6) D=$T/decoder_v6_BLIND.pt;; esac
  for set in real ham; do
    MODE=$2 $HOME/rfd4_venv/bin/python diag_decode_chir.py $D diag/${1}_${2}_$set.json $set 2>&1 | grep -E "SUMMARY|Traceback|Error" | cut -c1-420
  done
done
echo DIAG_DONE
