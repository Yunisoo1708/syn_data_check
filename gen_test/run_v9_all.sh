#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
bash run_diag_atoms.sh run7 $T/decoder_v9_BAL.pt
echo DIAG9_DONE
for m in G_ncw10 G2_pool G3_bal; do
  A=$T/out/$m/best; o=$T/gen_test/out/${m}_v9
  $HOME/rfd4_venv/bin/python gen_run.py ${m}_v9 $A $T/decoder_v9_BAL.pt $o "" 8 > $o.log 2>&1
  echo "$m v9 done $(date)"
done
echo V9_ALL_DONE
