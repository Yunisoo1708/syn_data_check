#!/bin/bash
# per-NCAA-position decoder-alone test on true latents for vanilla, run 3, run 5b, run 8 (diag_per_code.py)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test; mkdir -p diag
for pair in vanilla:bundle run3:$T/decoder_v5_493747.pt run5b:$T/decoder_v7b_COORD2.pt run8:$T/decoder_v10_BOND.pt; do
  n=${pair%%:*}; p=${pair#*:}
  $HOME/rfd4_venv/bin/python diag_per_code.py $p diag/percode_$n.json 2>&1 | grep -E "DONE|ERR|Traceback" | tail -3
  echo "percode $n done $(date)"
done
echo PERCODE_ALL_DONE
