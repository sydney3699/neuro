#!/bin/bash -l
#SBATCH --job-name=cmmerge
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/cole.sy/neuro/logs/cmmerge_%j.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/cmmerge_%j.err

# Reassemble the pair-parallel COMMOT chunks for one region into a single
# commot_<region>.h5ad. Concatenates every chunk's per-pair sender/receiver
# columns, recomputes the -total-total and per-pathway sums from the full pair
# set (deferred in the chunks), attaches obs + spatial, and writes the per-cell
# signaling niche used downstream. Run only AFTER all chunk array tasks succeed.
#
# Submit (V1 default):   sbatch commot_merge.sbatch
#        V2:             sbatch --export=ALL,REGION=v2 commot_merge.sbatch

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs

REGION="${REGION:-v1}"

echo "Host: $(hostname)  Region: ${REGION}  Start: $(date)"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/merge_commot.py \
    --region "$REGION" \
    --chunk-dir /scratch/cole.sy/neuro/results/commot \
    --outdir    /scratch/cole.sy/neuro/results/commot

echo "End: $(date)"
