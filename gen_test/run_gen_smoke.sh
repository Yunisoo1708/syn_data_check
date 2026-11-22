#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
$HOME/rfd4_venv/bin/python gen_run.py smoke_control none $T/decoder_v6_BLIND.pt $T/gen_test/out/smoke 7XAU_F 2 2>&1 | grep -v -i "warn" | tail -40
