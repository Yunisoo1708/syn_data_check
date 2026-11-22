#!/bin/bash
#SBATCH --job-name=pdg_cpu_klone
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/scrubbed/legia8/rosetta_score/pdg_%x_%j.log
# CPU-only (no --gres): free on Klone ckpt (idle nodes of other groups, may be preempted: --requeue). Paired Rosetta dG with matched relax, one position at a time.
# usage: sbatch --export=ALL,SLICE=0:20,DONE=laptop_partial.json slurm/paired_dg_cpu_klone.slurm
R=/gscratch/scrubbed/legia8/rosetta_score; cd $R
export PYTHONPATH=$R/pylib:$R
python3 paired_dg_matched.py ros_in real_paired_dg_relaxed.json out_klone_${SLURM_JOB_ID}.json 2 3 32 > run_klone_${SLURM_JOB_ID}.log 2>&1
