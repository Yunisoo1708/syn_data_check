#!/bin/bash
# FREE-SEQUENCE generation: the receptor pocket is fixed, the whole peptide (NCAA slots included) has NO given identity (spec_free.yaml); NCAA slots keep their token type
# from the input. Denoisers control, G, G3 x decoders run 8 (identity-conditioned) and run 4 (blind: trained with the identity hidden). 88 designs per run.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 GEN_SPEC=spec_free.yaml
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
for dn in control G_ncw10 G3_bal; do
  case $dn in control) A=none;; *) A=$T/out/$dn/best;; esac
  for dc in run8:$T/decoder_v10_BOND.pt blind:$T/decoder_v6_BLIND.pt; do
    n=${dc%%:*}; p=${dc#*:}; o=$T/gen_test/out/${dn}_free_$n
    $HOME/rfd4_venv/bin/python gen_run.py ${dn}_free_$n $A $p $o "" 8 > $o.log 2>&1; echo "${dn}_free_$n done $(date)"
  done
done
echo FREE_ALL_DONE
