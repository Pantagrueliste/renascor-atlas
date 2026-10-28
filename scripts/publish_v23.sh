#!/usr/bin/env bash
# Copy the v2.3 build from Hypatia to RQ/derived/atlas/v2.3/ (the file set of v2.2 plus the Latin hold-out tables of the geocoder, kind_overrides.tsv and orbis_latinus.tsv).
# Run on the Mac after slurm/v23_chain.sh has finished (rsync only; nothing is computed here except two small scripts).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R="${RATLAS_REMOTE:-hypatia}"
D=renascor-atlas-data
DEST="${1:-/Users/clem/Projects/Papers/RQ/derived/atlas/v2.3}"
V2=/Users/clem/Projects/Papers/RQ/derived/atlas/v2
mkdir -p "$DEST/geocode" "$DEST/audit_v2" "$DEST/stats" "$DEST/overrides"
RS="nice -n 19 rsync -a --partial"
$RS --exclude 'reconcile' --exclude '*/' "$R:$D/out_v2_3/all/" "$DEST/"                       # mentions, entities, *_local, collections, deviations ...
$RS "$R:$D/out_v2_3/reconcile/final/" "$DEST/"                                                # the reconcile tables, gate log, globe, report
$RS --exclude 'gn_index.parquet' "$R:$D/geocode_v2_3/" "$DEST/geocode/"
$RS "$R:$D/out_v2_3/audit_v2/" "$DEST/audit_v2/"
for s in $(ssh "$R" "ls $D/out_v2_3 | grep -v -e '^all$' -e '^reconcile$' -e '^audit_v2$'"); do
  mkdir -p "$DEST/stats/$s"; $RS "$R:$D/out_v2_3/$s/stats.json" "$DEST/stats/$s/" 2>/dev/null || true
done
cp "$REPO/data/licences.tsv" "$DEST/licences.tsv"
cp "$REPO/data/id_overrides.tsv" "$REPO/data/coord_overrides.tsv" "$REPO/data/geocode_rejects.tsv" "$REPO/data/kind_overrides.tsv" "$DEST/overrides/"
cp "$REPO/data/orbis_latinus.tsv" "$REPO/data/orbis_latinus.README.md" "$DEST/overrides/" 2>/dev/null || cp "$REPO/data/orbis_latinus.tsv" "$DEST/overrides/"
cp "$REPO/data/calibration_v2_1.json" "$DEST/calibration_v2_1.json"
cp "$V2/manual_review.tsv" "$V2/manual_review_effect.tsv" "$V2/manual_review_summary.tsv" "$V2/scan_extra.jsonl.gz" "$DEST/"
cd "$REPO"
nice -n 19 .venv/bin/python -I scripts/gate_counts.py --v2 "$DEST"
nice -n 19 .venv/bin/python -I scripts/compare_versions.py --v23 "$DEST" --out "$DEST/version_comparison.json"
echo "published to $DEST"
