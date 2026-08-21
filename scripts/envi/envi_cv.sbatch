#!/bin/bash -l
#SBATCH --job-name=envi_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/cole.sy/neuro/logs/envi_cv_%A_%a.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/envi_cv_%A_%a.err

# 5-fold leave-genes-out CV of the ENVI imputation: each array task retrains
# ENVI with one fold of shared panel genes withheld from the spatial side and
# scores imputed-vs-measured for those genes. Aggregate afterwards with:
#   conda run -n neuro python scripts/envi/aggregate_cv.py

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results/envi_cv

echo "Host: $(hostname)  Fold: ${SLURM_ARRAY_TASK_ID}  Start: $(date)"
nvidia-smi || true

TRAINING_STEPS="${TRAINING_STEPS:-10000}"
NUM_HVG="${NUM_HVG:-2048}"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/envi/run_envi_cv.py \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --n-folds 5 \
    --spatial /scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad \
    --snrna   /scratch/cole.sy/neuro/data/raw/snrna.h5ad \
    --outdir  /scratch/cole.sy/neuro/results/envi_cv \
    --training-steps "$TRAINING_STEPS" \
    --num-hvg "$NUM_HVG" \
    --celltype-key H2_annotation

echo "End: $(date)"
