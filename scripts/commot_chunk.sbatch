#!/bin/bash -l
#SBATCH --job-name=cmchunk
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=/scratch/cole.sy/neuro/logs/cmchunk_%A_%a.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/cmchunk_%A_%a.err

# PAIR-PARALLEL COMMOT over ALL cells of one region (no cell subsampling).
#
# Each array task keeps every cell and runs 1/NCHUNK of the L-R pairs, round-
# robin (run_commot.py --pair-chunk i/N: db_f.iloc[i::N], so slow pairs spread
# evenly), writing slim per-pair sender/receiver parquets. Reassemble afterward:
#   sbatch commot_merge.sbatch                       # V1
#   sbatch --export=ALL,REGION=v2 commot_merge.sbatch
#
# IMPORTANT: keep --array and NCHUNK in sync. --array is a static directive, so
# to change the chunk count edit BOTH the "#SBATCH --array=0-<N-1>" line above
# and NCHUNK below. Size N from the probe: N = total_pairs * sec_per_pair / 9000
# (targets ~2.5h/chunk; the 6h wall gives generous headroom).
#
# COMMOT is CPU-only, so this uses 'short' (2-day limit), no GPU.
#
# Submit (V1 default):   sbatch commot_chunk.sbatch
#        V2:             sbatch --export=ALL,REGION=v2 commot_chunk.sbatch

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results/commot

REGION="${REGION:-v1}"
NCHUNK="${NCHUNK:-32}"            # MUST match the --array range above (0..N-1)
COT_NITERMAX="${COT_NITERMAX:-10000}"

echo "Host: $(hostname)  Region: ${REGION}  Chunk: ${SLURM_ARRAY_TASK_ID}/${NCHUNK}  nitermax: ${COT_NITERMAX}  Start: $(date)"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/run_commot.py \
    --region "$REGION" \
    --pair-chunk "${SLURM_ARRAY_TASK_ID}/${NCHUNK}" \
    --cot-nitermax "$COT_NITERMAX" \
    --outdir /scratch/cole.sy/neuro/results/commot

echo "End: $(date)"
