#!/bin/bash -l
#SBATCH --job-name=cmchunk
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23
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
# and NCHUNK below.
#
# Sizing (measured, all 52k cells, nitermax=500): per-pair cost is heterogeneous,
# ~1080-1470s/pair (probe 8071285/8071286). At N=24 -> ~67 pairs/chunk; worst
# case ~67*1470s ~= 27h, comfortably under the 2-day wall. Round-robin spreads
# the expensive pairs so no single chunk is overloaded.
#
# MEMORY: COMMOT holds a cell x cell transport matrix per pair (~3 GB/pair at
# 52k cells), so running a whole chunk at once OOMs (68 pairs needed ~200-240 GB
# and killed the first launch). run_commot.py --pair-batch processes pairs a few
# at a time and frees them, capping peak at ~33 GB base + PAIR_BATCH*3 GB. At
# PAIR_BATCH=4 that's ~45 GB, so --mem=60G fits the plentiful 62 GB 'short'
# nodes (only a few nodes have >62 GB). PAIR_BATCH does NOT change results.
#
# COMMOT is CPU-only, so this uses 'short' (2-day limit), no GPU.
#
# Submit (V1 default):   sbatch commot_chunk.sbatch
#        V2:             sbatch --export=ALL,REGION=v2 commot_chunk.sbatch

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results/commot

REGION="${REGION:-v1}"
NCHUNK="${NCHUNK:-24}"            # MUST match the --array range above (0..N-1)
COT_NITERMAX="${COT_NITERMAX:-500}"   # nitermax verified irrelevant at 52k scale; 500 is no slower
PAIR_BATCH="${PAIR_BATCH:-4}"    # pairs processed at once; caps peak memory (see MEMORY note above)

echo "Host: $(hostname)  Region: ${REGION}  Chunk: ${SLURM_ARRAY_TASK_ID}/${NCHUNK}  nitermax: ${COT_NITERMAX}  pair_batch: ${PAIR_BATCH}  Start: $(date)"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/commot/run_commot.py \
    --region "$REGION" \
    --pair-chunk "${SLURM_ARRAY_TASK_ID}/${NCHUNK}" \
    --pair-batch "$PAIR_BATCH" \
    --cot-nitermax "$COT_NITERMAX" \
    --outdir /scratch/cole.sy/neuro/results/commot

echo "End: $(date)"
