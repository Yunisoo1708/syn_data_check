#!/bin/bash
# FREE-CHOICE generation (Results.MD 81): the receptor pocket is fixed and the identity of the peptide's NCAA tokens is HIDDEN (box pipeline patched with patch_hide_nc.py,
# HIDE_NC_IDENTITY_CHAIN_PREFIX=P; spec_free.yaml has no C_SEQ on the peptide). The NCAA token TYPE is kept. The NCAA identity then comes from the denoiser's own sequence readout
# (condition_decoder_on_sequence_prediction is on by default) restricted to NCAA classes, and decoder run 8 builds the atoms. Denoisers control, G3, S1 (readout trained on hidden NCAAs). 11 pockets x 8 designs.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 GEN_SPEC=spec_free.yaml HIDE_NC_IDENTITY_CHAIN_PREFIX=P
export PYTHONPATH=$T/patch:$T/code
cd $T/gen_test
for dn in ${DNS:-control G3_bal S1_readout}; do
  case $dn in control) A=none;; *) A=$T/out/$dn/best;; esac
  o=$T/gen_test/out/${dn}_choice_run8
  $HOME/rfd4_venv/bin/python gen_run.py ${dn}_choice_run8 $A $T/decoder_v10_BOND.pt $o "" 8 > $o.log 2>&1; echo "${dn}_choice_run8 done $(date)"
done
echo CHOICE_ALL_DONE
