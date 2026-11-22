#!/bin/bash
# re-run the free-sequence combinations that did not finish (control+blind, G+run8, G+blind, G3+run8), after the denoise-then-decode test (one GPU)
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 GEN_SPEC=spec_free.yaml
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
while ! grep -q DD_ALL_DONE run_dd.log 2>/dev/null; do sleep 30; done
for combo in control:blind G_ncw10:run8 G_ncw10:blind G3_bal:run8; do
  dn=${combo%%:*}; d=${combo#*:}
  case $dn in control) A=none;; *) A=$T/out/$dn/best;; esac
  case $d in run8) p=$T/decoder_v10_BOND.pt;; blind) p=$T/decoder_v6_BLIND.pt;; esac
  o=$T/gen_test/out/${dn}_free_$d; rm -rf $o
  $HOME/rfd4_venv/bin/python gen_run.py ${dn}_free_$d $A $p $o "" 8 > $o.log 2>&1; echo "${dn}_free_$d done $(date) $(ls $o/inference_samples/*/*_generated.cif 2>/dev/null | wc -l) designs"
done
echo FREE_MISSING_DONE
