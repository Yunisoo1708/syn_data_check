#!/bin/bash
# usage: run_diag_atoms.sh NAME DECODER.pt   -> diag/atoms_NAME_{real,ham}.json (identity given)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
for set in real ham; do MODE=given $HOME/rfd4_venv/bin/python diag_decode_atoms.py $2 diag/atoms_$1_$set.json $set 2>&1 | grep -E "SUMMARY|Traceback|Error" | sed -E 's/"decoder": "[^"]*", //' | cut -c1-600; done
echo ATOMS_$1_DONE
