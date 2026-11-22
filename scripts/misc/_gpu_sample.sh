#!/bin/bash
# sample GPU memory/util of job 493150 every 6 s for 18 samples and print the step counter
for i in $(seq 1 18); do
  srun --jobid=493150 --overlap -N1 -n1 nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' '
  tail -n 1 /gpfs/projects/stf/legia8/RFD4_189/dn_F_H200_493150.train.log | cut -c1-60
  sleep 3
done
