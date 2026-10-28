#!/bin/bash
# v2: the nine checks of the audit as blocking checks on the final tables (src/ratlas/audit_v2.py), then the real-data pytest.
#   sbatch --output=$HOME/renascor-atlas-data/logs/%x-%j.out slurm/audit_v2.sbatch [extra args for `ratlas audit v2`]
# reads  out_v2/all (mentions, *_local), out_v2/reconcile/final (the tables), reconcile_v2 (authorities), raw (SOURCES.tsv, manifests),
#        geonames/allCountries.zip (check 4: one pass)
# writes out_v2/audit_v2/ (check<N>_<name>.tsv = the failures, audit_v2_summary.json, gn_final_dist.tsv)
#SBATCH --job-name=ratlas-audit2
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -uo pipefail
D="$HOME/renascor-atlas-data"
cd "$HOME/renascor-atlas"
nice -n 10 .venv/bin/python -I -m ratlas audit v2 --all "$D/out_v2/all" --final "${FINAL:-$D/out_v2/reconcile/final}" --rdir "$D/reconcile_v2" \
  --raw "$D/raw" --out "$D/out_v2/audit_v2" --geonames-zip "$D/geonames/allCountries.zip" "$@"
rc=$?
echo "audit exit code $rc"
RATLAS_AUDIT_V2=1 RATLAS_AUDIT_ALL="$D/out_v2/all" RATLAS_AUDIT_FINAL="${FINAL:-$D/out_v2/reconcile/final}" RATLAS_AUDIT_RDIR="$D/reconcile_v2" \
  RATLAS_AUDIT_RAW="$D/raw" RATLAS_AUDIT_OUT="$D/out_v2/audit_v2" RATLAS_AUDIT_GEONAMES="$D/geonames/allCountries.zip" \
  nice -n 10 .venv/bin/python -I -m pytest tests/test_audit_v2.py -x -q -p no:cacheprovider
exit $(( rc != 0 ))
