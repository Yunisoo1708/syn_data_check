#!/bin/bash
cd ~/rfd4_trial
for p in $(ps -eo pid,args | grep -E "run_eval_real_gpu1|eval_one_real.py C_long" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 3
nohup bash run_eval_real_gpu0.sh > out/eval_real0.log 2>&1 &
sleep 2; ps -eo pid,args | grep -E "eval_one_real|run_eval_real" | grep -v grep | cut -c1-140
