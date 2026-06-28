#!/bin/bash -l
#SBATCH --job-name=envi_fb080
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cole.sy/neuro/logs/envi_%j.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/envi_%j.err

# ENVI integration of FB080 V1/V2 MERFISH spatial data with the paired snRNA-seq
# reference on a single V100. Adjust --training-steps / --num-hvg / --celltype-key below.

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results

echo "Host: $(hostname)   Start: $(date)"
nvidia-smi || true

# Override at submit time, e.g.: sbatch --export=ALL,TRAINING_STEPS=1000 envi_gpu.sbatch
TRAINING_STEPS="${TRAINING_STEPS:-10000}"
NUM_HVG="${NUM_HVG:-2048}"
OUTDIR="${OUTDIR:-/scratch/cole.sy/neuro/results/envi_FB080}"
CELLTYPE_KEY="${CELLTYPE_KEY:-H2_annotation}"
echo "TRAINING_STEPS=$TRAINING_STEPS  NUM_HVG=$NUM_HVG  OUTDIR=$OUTDIR  CELLTYPE_KEY=$CELLTYPE_KEY"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/run_envi.py \
    --spatial /scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad \
    --snrna   /scratch/cole.sy/neuro/data/raw/snrna.h5ad \
    --outdir  "$OUTDIR" \
    --training-steps "$TRAINING_STEPS" \
    --num-hvg "$NUM_HVG" \
    --celltype-key "$CELLTYPE_KEY"

echo "End: $(date)"
