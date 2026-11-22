#!/bin/bash
#SBATCH --job-name=ncf_cpu
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/projects/stf/legia8/NCAA_dataset/ncflow_pilot/cpu_%x_%A_%a.log
# CPU-only NCFlow pilot generator (no GPU requested). usage: sbatch --array=S1,S2 --export=ALL,ITEMS=pilot_items_v3_dform.json,NSHARD=32,TAG=v3dform ncf_cpu_tillicum.slurm
N=/gpfs/projects/stf/legia8/NCAA_dataset
cd $N/ncflow_pilot
export PYTHONPATH=$N/ncflow_libs:$N/ncflow_pilot CPATH=$N/pyinc PYTHONUNBUFFERED=1 NCF_GEO=$N/geo_mols OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CAP=${CAP:-12} MAXB=3
SH=${SLURM_ARRAY_TASK_ID:-0}
OUT=out2/${TAG}_n${NSHARD}_s${SH}.pkl
[ -f "$OUT.done" ] && { echo "shard $SH already done"; exit 0; }
echo "node $(hostname) cpus $SLURM_CPUS_PER_TASK shard $SH/$NSHARD tag $TAG start $(date)"
$N/venv_boltz2/bin/python ncf_pilot.py out2/rl_syn.pth $TAG $OUT 4 $SH $NSHARD $ITEMS && touch $OUT.done
echo "end $(date)"
