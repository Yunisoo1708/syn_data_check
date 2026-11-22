#!/bin/bash
# usage: run_refold.sh INPUT_DIR OUTPUT_DIR   (AF3, single GPU 0, template receptor, 3 diffusion samples)
T=$HOME/rfd4_trial/refold
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
$HOME/miniconda3/envs/af3/bin/python $HOME/tools/alphafold3/run_alphafold.py --input_dir=$1 --output_dir=$2 --model_dir=$HOME/af3_models --run_data_pipeline=false --run_inference=true --num_diffusion_samples=3 --num_recycles=10 2>&1 | grep -vE "^I[0-9]{4}|^W[0-9]{4}|Skipping|xla|XLA" | tail -40
echo REFOLD_DONE
