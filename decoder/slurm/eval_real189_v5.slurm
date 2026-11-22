#!/bin/bash
#SBATCH --job-name=ae_real189
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:40:00
#SBATCH --output=/gpfs/projects/stf/legia8/RFD4_189/eval_v5/ae_real189_%j.log
# Real-pose test (2,765 positions, 189 classes) of the untrained bundle, run 1 (v3), run 2 (v4) and run 3 (v5, mirror-image D-peptides): about 0.05 USD on a MIG slice
B=/gpfs/projects/stf/legia8; R=$B/RFD4_189; mkdir -p $R/eval_v5; cd $R/eval_v5
export RFD_BUNDLE=$R/bundle_189 BENCH=$R/bench189 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 RFD4_VOCAB_YAML=$R/aa_rna_dna_glycan_ncaa_ext_v3.yaml PYTHONPATH=$R:$B/RFD4_fine_tune
$HOME/rfd4_venv/bin/python $R/eval_real_189_v3.py v3=$R/decoder_v3_485816.pt v4=$R/decoder_v4_488524.pt v5=$R/decoder_v5_493747.pt 2>&1 | grep -v -E "Environment variable|^\(|cuEquiv|warn" | tail -n 120
echo EXIT $?
