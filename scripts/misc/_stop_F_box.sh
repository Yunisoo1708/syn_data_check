#!/bin/bash
# stop trial F on the box cleanly (SIGTERM -> the script saves 'last' on exit); best/ckpt are already saved at each eval
cd ~/rfd4_trial
for p in $(ps -eo pid,args | grep -E "ncaa_fine_tuning_v2.py .*F_pool2|run_denoiser_trial_F" | grep -v grep | awk '{print $1}'); do echo "kill $p"; kill -TERM $p; done
sleep 25
ps -eo pid,args | grep -E "ncaa_fine_tuning|run_denoiser" | grep -v grep | cut -c1-120
echo ---; ls out/F_pool2; tail -n 2 out/F_pool2_bs1.log | cut -c1-100
grep -E "^eval step" out/F_pool2_bs1.log | sed -E "s/terms.*baseline [0-9.]+ //" | cut -c1-120 | tail -n 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
