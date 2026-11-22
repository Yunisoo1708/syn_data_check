#!/bin/bash
# generation test: 4 denoisers x 2 decoders, 11 real complexes, 8 designs each (same seed)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
for dec in v6 v5; do
  case $dec in v5) D=$T/decoder_v5_493747.pt;; v6) D=$T/decoder_v6_BLIND.pt;; esac
  for m in control G_ncw10 G2_pool G3_bal; do
    case $m in control) A=none;; *) A=$T/out/$m/best;; esac
    o=$T/gen_test/out/${m}_$dec
    $HOME/rfd4_venv/bin/python gen_run.py ${m}_$dec $A $D $o "" 8 > $o.log 2>&1 && touch $o.done
    echo "$m $dec done $(date)"
  done
done
echo GEN_ALL_DONE
