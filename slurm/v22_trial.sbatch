#!/bin/bash
# v2.2 trial: build2 final with the v2.2 code on the v2.1 inputs (extraction, geocoder results), written to scratch_v22/ (nothing of v2.1 is touched).
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/v22_trial.sbatch
#SBATCH --job-name=ratlas-v22-trial
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
D="$HOME/renascor-atlas-data"; S="$D/scratch_v22"; RD="$S/rdir"
mkdir -p "$RD" "$S/final"
cp "$D"/reconcile_v2_1/* "$RD/"
cd "$HOME/renascor-atlas"
cp data/id_overrides.tsv data/coord_overrides.tsv data/geocode_rejects.tsv data/casebooks_place_map.tsv "$RD/"
export RATLAS_DATA="$D" RATLAS_OUT="$D/out_v2_1"
nice -n 10 .venv/bin/python -I -m ratlas reconcile build2 --v2 "$D/out_v2_1/all" --rdir "$RD" --dest "$S/final" --tgn-dir "$D/tgn" \
  --casebooks-map "$RD/casebooks_place_map.tsv" --namecheck "$RD/namecheck_local.tsv" --geocode "$D/geocode_v2_1/geocode_places.parquet" \
  --geocode-verify "$D/geocode_v2_1/geocode_verify.tsv"

# the nine checks of the audit on the trial tables (their own output folder)
nice -n 10 .venv/bin/python -I -m ratlas audit v2 --all "$D/out_v2_1/all" --final "$S/final" --rdir "$RD" --raw "$D/raw" --out "$S/audit_v2" \
  --geonames-zip "$D/geonames/allCountries.zip" --tgn-dir "$D/tgn" || echo "audit exit code $?"
