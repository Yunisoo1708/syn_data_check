#!/bin/bash
#SBATCH --job-name=ae_eval_real
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:40:00
B=/gpfs/projects/stf/legia8
cd $B/RFD4_fine_tune
export RFD_BUNDLE=$B/RFD4_fine_tune/models/RFD4_bundle_176 BENCH=$B/NCAA_dataset/real_benchmark HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
$HOME/rfd4_venv/bin/python eval_real_autoencoder.py run1=$B/RFD4_fine_tune/weights/decoder/best_decoder_run1.pt run2=$B/RFD4_fine_tune/weights/decoder/best_decoder_run2.pt 2>&1 | grep -v -E "Environment variable|^\(|cuEquiv|fork|self.pid|Warning"
echo EXIT $?
