#!/bin/bash
# probe where the NCAA identity enters generation (box, GPU 0): bash run_probe_identity.sh [EXAMPLE] [SPEC]
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
export HIDE_NC_IDENTITY_CHAIN_PREFIX=${HIDE_PREFIX:-}
timeout 900 $HOME/rfd4_venv/bin/python probe_identity.py ${1:-7PIU_P} ${2:-spec_free.yaml} 2>&1 | grep -v -i warn | tail -40
