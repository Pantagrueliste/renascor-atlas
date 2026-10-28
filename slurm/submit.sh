#!/usr/bin/env bash
# Run on Hypatia (login node: only submits): slurm/submit.sh slug1 slug2 ...
# Env overrides: PARTITION (default debug), TIME (default 00:30:00), MEM (16G), CPUS (8, max 8), LIMIT (first N files),
# CONCURRENT (max simultaneous array tasks, default 2).
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: $0 slug..." >&2; exit 2; }
mkdir -p "$HOME/renascor-atlas-data/logs" "$HOME/renascor-atlas-data/out"
N=$#
CPUS="${CPUS:-8}"; [ "$CPUS" -le 8 ] || CPUS=8
cd "$HOME/renascor-atlas"
SLUGS="$*" sbatch --export=ALL,SLUGS="$*",LIMIT="${LIMIT:-}" \
  --partition="${PARTITION:-debug}" --time="${TIME:-00:30:00}" --mem="${MEM:-16G}" --cpus-per-task="$CPUS" \
  --output="$HOME/renascor-atlas-data/logs/%x-%A_%a.out" \
  --array="0-$((N-1))%${CONCURRENT:-2}" slurm/extract.sbatch
