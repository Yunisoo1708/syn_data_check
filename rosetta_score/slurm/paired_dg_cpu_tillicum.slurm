#!/bin/bash
#SBATCH --job-name=pdg_cpu
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/rosetta_score/pdg_%x_%j.log
# CPU-only (no --gres): free on Tillicum (sbatch shows COST $0.00). Paired Rosetta dG with matched relax, one position at a time.
# usage: sbatch --export=ALL,SLICE=0:20,DONE=laptop_partial.json slurm/paired_dg_cpu_tillicum.slurm
R=/gpfs/projects/stf/legia8/rosetta_score; cd $R
export PYTHONPATH=$R/pylib:$R
python3 paired_dg_matched.py ros_in real_paired_dg_relaxed.json out_tillicum_${SLURM_JOB_ID}.json 2 3 32 > run_tillicum_${SLURM_JOB_ID}.log 2>&1
