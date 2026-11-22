#!/bin/bash
# WSL: pull the free-choice runs from the box (copy only) and analyse them with analyze_free.py (writes diag/free.json; saved as diag/free_choice.json, the old identity-given result is restored)
G=/mnt/c/Research/RFD4_fine_tune/gen_test; cd $G; mkdir -p out_local
for n in control G3_bal S1_readout; do
  scp -q -r ${BOX:-user@box}:~/rfd4_trial/gen_test/out/${n}_choice_run8 out_local/ && echo PULLED_${n} $(ls out_local/${n}_choice_run8/inference_samples | wc -l)
done
cp diag/free.json diag/free_identity_given_backup.json
source ~/rfd4_venv/bin/activate
python analyze_free.py control_choice_run8=out_local/control_choice_run8 G3_choice_run8=out_local/G3_bal_choice_run8 S1_choice_run8=out_local/S1_readout_choice_run8 2>&1 | tail -5 | cut -c1-400
cp diag/free.json diag/free_choice.json; cp diag/free_identity_given_backup.json diag/free.json
