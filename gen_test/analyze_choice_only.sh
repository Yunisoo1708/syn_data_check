#!/bin/bash
G=/mnt/c/Research/RFD4_fine_tune/gen_test; cd $G; cp diag/free.json diag/free_identity_given_backup.json; source ~/rfd4_venv/bin/activate
python analyze_free.py control_choice_run8=out_local/control_choice_run8 G3_choice_run8=out_local/G3_bal_choice_run8 S1_choice_run8=out_local/S1_readout_choice_run8 2>&1 | tail -6 | cut -c1-420
cp diag/free.json diag/free_choice.json; cp diag/free_identity_given_backup.json diag/free.json
