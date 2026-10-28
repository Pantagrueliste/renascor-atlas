#!/bin/bash
# Scan of the re-harvested folders (originals_2026-10-17) into ~/renascor-atlas-data/scan_extra.jsonl.gz
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/scan_extra.sbatch
#SBATCH --job-name=ratlas-scan-extra
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
P=originals_2026-10-17
SCAN_WORKERS=8 nice -n 10 .venv/bin/python -I scripts/scan_extra.py "$D/raw" "$D/scan_extra.jsonl" \
  $P/abacus $P/bullinger $P/cavriana $P/dracor_eng $P/kaiserin_eleonora_magdalena $P/pceec2 $P/qhod_sema $P/rup
gzip -f "$D/scan_extra.jsonl"
ls -la "$D/scan_extra.jsonl.gz"
