#!/bin/bash
# usage: run_diag_noise.sh NAME DECODER.pt [CAJ]  -> diag/noise_NAME.json
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
CAJ=${3:-0} $HOME/rfd4_venv/bin/python diag_noise.py $2 diag/noise_$1.json real 2>&1 | grep -E "SIGMA|emit|Traceback|ERR" | cut -c1-300
echo NOISE_$1_DONE
