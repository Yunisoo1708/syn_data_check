#!/bin/bash
# usage: run_af3_iptm.sh <jobs_dir> <out_dir> <gpu> [num_samples=2]   (on the 2-GPU box; AF3 env, single-sequence, no data pipeline)
J=$1; O=$2; GPU=$3; NS=${4:-2}
mkdir -p $O ~/ncaa_fold/jaxcache
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false ~/miniconda3/envs/af3/bin/python ~/tools/alphafold3/run_alphafold.py \
  --input_dir=$J --output_dir=$O --model_dir=$HOME/af3_models --run_data_pipeline=false --run_inference=true \
  --num_diffusion_samples=$NS --num_recycles=10 --jax_compilation_cache_dir=$HOME/ncaa_fold/jaxcache
