#!/bin/bash
# fresh batch for the best-of-N clash filter test: G3 and G with decoder run 8, seed 777 (the first test used seed 1234), 8 designs per pocket
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 GEN_SEED=777
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
for m in G3_bal G_ncw10; do
  o=$T/gen_test/out/${m}_v10_s777
  $HOME/rfd4_venv/bin/python gen_run.py ${m}_v10_s777 $T/out/$m/best $T/decoder_v10_BOND.pt $o "" 8 > $o.log 2>&1
  echo "$m s777 done $(date)"
done
echo S777_DONE
