#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
o=$T/gen_test/out/G4_clash_v10
$HOME/rfd4_venv/bin/python gen_run.py G4_clash_v10 $T/out/G4_clash/best $T/decoder_v10_BOND.pt $o "" 8 > $o.log 2>&1
echo G4_DONE
