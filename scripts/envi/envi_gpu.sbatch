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
# LR_DATABASE=cellchat force-includes CellChatDB L-R genes (needed for COMMOT);
# OUTDIR defaults to a separate dir so the HVG-only run is preserved for comparison.
TRAINING_STEPS="${TRAINING_STEPS:-10000}"
NUM_HVG="${NUM_HVG:-2048}"
LR_DATABASE="${LR_DATABASE:-cellchat}"
OUTDIR="${OUTDIR:-/scratch/cole.sy/neuro/results/envi_FB080_lr}"
CELLTYPE_KEY="${CELLTYPE_KEY:-H2_annotation}"
echo "TRAINING_STEPS=$TRAINING_STEPS  NUM_HVG=$NUM_HVG  LR_DATABASE=$LR_DATABASE  OUTDIR=$OUTDIR  CELLTYPE_KEY=$CELLTYPE_KEY"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/envi/run_envi.py \
    --spatial /scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad \
    --snrna   /scratch/cole.sy/neuro/data/raw/snrna.h5ad \
    --outdir  "$OUTDIR" \
    --training-steps "$TRAINING_STEPS" \
    --num-hvg "$NUM_HVG" \
    --lr-database "$LR_DATABASE" \
    --celltype-key "$CELLTYPE_KEY"

echo "End: $(date)"
