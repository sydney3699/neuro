#!/bin/bash -l
#SBATCH --job-name=envi_niche_h2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/cole.sy/neuro/logs/envi_%j.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/envi_%j.err
set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results
echo "Host: $(hostname)   Start: $(date)"; nvidia-smi || true
conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/run_envi.py \
    --spatial /scratch/cole.sy/neuro/data/processed/FB080_O1b_v1v2_envi_input.h5ad \
    --snrna   /scratch/cole.sy/neuro/data/raw/snrna.h5ad \
    --outdir  /scratch/cole.sy/neuro/results/envi_FB080_niche_H2 \
    --training-steps 10000 --num-hvg 2048 --celltype-key H2_annotation
echo "End: $(date)"
