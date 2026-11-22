#!/bin/bash
# Denoiser (flow LoRA) trial on the box RTX 5060 Ti (GPU 0): A = all clean complexes, B = first batch only. Waits for the copy, pauses the general batch after its current shard, falls back to batch size 1 on out-of-memory.
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
PY=$HOME/rfd4_venv/bin/python
mkdir -p $T/out
echo "waiting for the copy $(date)"
until [ -f $T/bundle_189/model.safetensors ] && [ "$(stat -c %s $T/bundle_189/model.safetensors)" -ge 1743000000 ] && [ "$(ls $T/train_all_cif_clean | wc -l)" -ge 1429 ] && ! pgrep -f "[r]sync" > /dev/null; do sleep 20; done
echo "copy complete $(date); pausing the general batch on GPU 0 after its current shard"
pkill -f "[r]un_box_gen.sh 80"
while pgrep -f "[n]cf_pilot.py.* 4 8[0-9] 128" > /dev/null || pgrep -f "[n]cf_pilot.py.* 4 9[0-9] 128" > /dev/null; do sleep 20; done
run() {  # TAG FILTER SECONDS
  for BS in 1; do
    echo "== $1 bs $BS $(date)"
    $PY $T/patch/ncaa_fine_tuning_v2.py $T/bundle_189 $T/train_all_cif_clean $T/out/$1 --groups $T/complex_groups.json ${2:+--filter $2} --steps 1000 --bs $BS --inner 5 --eval-every 100 --eval-batches 14 --patience 4 --max-seconds $3 > $T/out/$1_bs$BS.log 2>&1
    grep -qE "out of memory|GPU mapping|device not ready|CUDA driver error" $T/out/$1_bs$BS.log && continue
    break
  done
}
run A_all "" 14400
run B_first _ncf 7200
echo ALLDONE $(date)
