#!/bin/bash
T=$HOME/rfd4_trial
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256 RFD4_VOCAB_YAML=$T/patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$T/patch:$T/code
mkdir -p $T/out/eval_nc
# quick smoke test: first 3 real complexes
sed -e 's#^res = {}; t0 = time.time()#files = files[:3]; res = {}; t0 = time.time()#' $T/eval_one_nc.py > $T/eval_one_nc_smoke.py
$HOME/rfd4_venv/bin/python $T/eval_one_nc_smoke.py control none $T/out/eval_nc/smoke.json real 2>&1 | grep -E "vocab|control|Traceback|Error|rror:" | tail -n 8
python3 - <<PY
import json
d=json.load(open("$T/out/eval_nc/smoke.json"))["results"]
for f,v in d.items(): print(f, str(v)[:700])
PY
