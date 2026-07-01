#!/bin/bash -l
#SBATCH --job-name=cmprobe
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/cole.sy/neuro/logs/cmprobe_%j.out
#SBATCH --error=/scratch/cole.sy/neuro/logs/cmprobe_%j.err

# CALIBRATION PROBE for the pair-parallel COMMOT run.
#
# Runs ONE pair-chunk (index 0 of PROBE_N) over ALL cells of one region, timing
# the spatial_communication call so we can measure real per-pair cost at full
# (~52k-cell) scale -- the number the earlier cmcalib run never produced because
# it hit a 45-min wall. Divide the reported "done in Ns" by the chunk's pair
# count to get seconds/pair, then size the array N in commot_chunk.sbatch.
#
# Writes to a SEPARATE probe outdir so it never mixes with the real chunk set.
# COMMOT is CPU-only (POT/scipy), so this runs on 'short' (2-day limit), no GPU.
#
# Submit (V1 default):   sbatch commot_probe.sbatch
#        V2:             sbatch --export=ALL,REGION=v2 commot_probe.sbatch

set -euo pipefail
mkdir -p /scratch/cole.sy/neuro/logs /scratch/cole.sy/neuro/results/commot_probe

REGION="${REGION:-v1}"
PROBE_N="${PROBE_N:-64}"          # chunk 0 of 64 ~= 25 representative (round-robin) pairs
COT_NITERMAX="${COT_NITERMAX:-10000}"

echo "Host: $(hostname)  Region: ${REGION}  Probe chunk: 0/${PROBE_N}  nitermax: ${COT_NITERMAX}  Start: $(date)"

conda run --no-capture-output -n neuro python -u /home/cole.sy/neuro/scripts/run_commot.py \
    --region "$REGION" \
    --pair-chunk "0/${PROBE_N}" \
    --cot-nitermax "$COT_NITERMAX" \
    --outdir /scratch/cole.sy/neuro/results/commot_probe

echo "End: $(date)"
echo "NEXT: seconds_per_pair = (done-in seconds) / (pairs in chunk, printed above);"
echo "      target ~2-3h/chunk -> N = total_pairs * seconds_per_pair / 9000"
