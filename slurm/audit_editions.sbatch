#!/bin/bash
# v2: the per-edition checks of the audit on a fresh sample (3 source files and the 10 top places of each of the 78 editions):
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out [--dependency=afterok:<audit job>] slurm/audit_editions.sbatch
# needs out_v2/audit_v2/gn_final_dist.tsv (slurm/audit_v2.sbatch, check 4); writes out_v2/audit_v2/editions/
#SBATCH --job-name=ratlas-audited
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas audit editions --all "$D/out_v2/all" --final "${FINAL:-$D/out_v2/reconcile/final}" --raw "$D/raw" \
  --out "$D/out_v2/audit_v2/editions" --rdir "$D/reconcile_v2" --dist "$D/out_v2/audit_v2/gn_final_dist.tsv" "$@"
