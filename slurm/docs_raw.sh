#!/usr/bin/env bash
# On Hypatia (login node: only submits): slurm/docs_raw.sh slug1 slug2 ...   (env: SAMPLE=N, RAWSUFFIX=_sample, TIME, MEM, CONCURRENT,
#   DOCS_DIR=directory that holds keys.parquet and receives raw<suffix>/; default ~/renascor-atlas-data/docs_cards)
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: $0 slug..." >&2; exit 2; }
mkdir -p "$HOME/renascor-atlas-data/logs" "${DOCS_DIR:-$HOME/renascor-atlas-data/docs_cards}"
cd "$HOME/renascor-atlas"
SLUGS="$*" sbatch --export=ALL,SLUGS="$*",SAMPLE="${SAMPLE:-0}",RAWSUFFIX="${RAWSUFFIX:-}",DOCS_DIR="${DOCS_DIR:-}" \
  --time="${TIME:-00:30:00}" --mem="${MEM:-16G}" \
  --output="$HOME/renascor-atlas-data/logs/%x-%A_%a.out" \
  --array="0-$(($#-1))%${CONCURRENT:-4}" slurm/docs_raw.sbatch
