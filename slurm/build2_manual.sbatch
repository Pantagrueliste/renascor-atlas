#!/bin/bash
# v2 final build with the manual-review overrides, on a frozen copy of the inputs of the v2 final build (2026-10-17):
#   ~/renascor-atlas-data/snapshot_manual/{all,reconcile_v2,geocode_v2}   (copied from RQ/derived/atlas/v2 and reconcile_v2)
# so that a re-extraction running in parallel cannot change the inputs. Output: ~/renascor-atlas-data/out_v2/reconcile/manual_review
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/build2_manual.sbatch [extra build2 arguments]
#SBATCH --job-name=ratlas-build2-manual
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"
S="$D/snapshot_manual"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas reconcile build2 --v2 "$S/all" --rdir "$S/reconcile_v2" --dest "$D/out_v2/reconcile/manual_review" \
  --namecheck "$S/reconcile_v2/namecheck_local.tsv" --geocode "$S/geocode_v2/geocode_places.parquet" \
  --geocode-verify "$S/geocode_v2/geocode_verify.tsv" "$@"
