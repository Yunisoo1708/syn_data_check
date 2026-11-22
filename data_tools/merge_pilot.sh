#!/bin/bash
# merge all 48 pilot shards (laptop out2/ and Klone out2/klone/) -> stats json, swaps json, figure data. usage: bash merge_pilot.sh   (WSL)
cd /mnt/c/Research/ncflow_ref/ncflow/out2
P=/home/yunisoo/miniconda3/envs/rfd3_env/bin/python
n=$(ls pilot_rlsyn_n48_s*.pkl klone/pilot_rlsyn_n48_s*.pkl 2>/dev/null | sed 's/.*_s//' | sort -un | wc -l)
echo "distinct shards: $n of 48"
[ "$n" = 48 ] || { echo "NOT ALL SHARDS PRESENT"; exit 1; }
F="pilot_rlsyn_n48_s*.pkl klone/pilot_rlsyn_n48_s*.pkl"
$P /mnt/c/Research/RFD4_fine_tune/data_tools/pilot_stats.py /mnt/c/Research/RFD4_fine_tune/data/lists/pilot_all_stats.json $F 2>&1 | grep -v -i warn
$P /mnt/c/Research/RFD4_fine_tune/data_tools/pilot_metrics.py /mnt/c/Research/RFD4_fine_tune/data/lists/pilot_all_swaps.json $F 2>&1 | grep -v -i warn
