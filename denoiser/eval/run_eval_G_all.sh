#!/bin/bash
# waits for trial G to end, then scores G_ncw10 (best weights) on every test (box GPU 0), then the denoise-then-decode test with the run 3 decoder (decoder_v5_493747.pt) for the control and all nine trained denoisers
T=$HOME/rfd4_trial
until grep -q "^G end" $T/out/G_launcher.log; do sleep 20; done
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python; A=$T/out/G_ncw10/best; n=G_ncw10
mkdir -p $T/out/eval $T/out/eval_real $T/out/eval_nc $T/out/eval_dec $T/out/eval_dec5
[ -f $T/out/eval/$n.json ] || $PY $T/eval_one.py $n $A $T/out/eval/$n.json 2>&1 | grep -E "Traceback|rror:" | tail -n 2
[ -f $T/out/eval_real/$n.json ] || $PY $T/eval_one_real.py $n $A $T/out/eval_real/$n.json 2>&1 | grep -E "Traceback|rror:" | tail -n 2
for set in real heldout ham; do [ -f $T/out/eval_nc/${set}_$n.json ] || $PY $T/eval_one_nc.py $n $A $T/out/eval_nc/${set}_$n.json $set 2>&1 | grep -E "Traceback|rror:" | tail -n 2; done
for set in real heldout ham; do
  o=$T/out/eval_dec/${set}_$n.json; [ -f $o ] && continue
  if [ $set = ham ]; then TS=0.1 NCLIST=$T/out/eval_dec/ham_nclist.json $PY $T/eval_one_dec.py $n $A $o $set $T/decoder_v4_488524.pt 2>&1 | grep -E "Traceback|rror:" | tail -n 2
  else TS=0.05,0.15,0.3 $PY $T/eval_one_dec.py $n $A $o $set $T/decoder_v4_488524.pt 2>&1 | grep -E "Traceback|rror:" | tail -n 2; fi
done
echo EVAL_G_DONE
