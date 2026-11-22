#!/bin/bash
#SBATCH --job-name=mirror
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/mirror_%j.log
# CPU-only (no GPU requested = no charge): mirror-image D-peptides from the decoder pool, 14,000 peptides
B=/gpfs/projects/stf/legia8
cd $B/RFD4_189
$B/NCAA_dataset/venv_boltz2/bin/python mirror_pool.py $B/NCAA_dataset/training_pool_189 $B/RFD4_189/pool_mirror3 14000
