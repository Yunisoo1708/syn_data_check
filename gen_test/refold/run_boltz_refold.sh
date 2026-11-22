#!/bin/bash
# usage: run_boltz_refold.sh IN_DIR OUT_DIR   (Boltz-2, GPU 0, 3 diffusion samples, forced receptor template)
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
$HOME/ncaa_fold/venv_boltz/bin/boltz predict $1 --out_dir $2 --devices 1 --accelerator gpu --diffusion_samples 3 --output_format mmcif --num_workers 2 --use_potentials --no_kernels 2>&1 | grep -v -E "Warning|warn|^\s*$" | tail -25
echo BOLTZ_DONE
