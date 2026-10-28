#!/usr/bin/env bash
# Copy the reports of the external-data / external-model work from Hypatia to the RQ derived folder (tables and counts only: no text, no weights).
#   scripts/ner_external_pull.sh
set -euo pipefail
REMOTE="${RATLAS_REMOTE:-hypatia}"
DEST="${NER_DEST:-/Users/clem/Projects/Papers/RQ/derived/atlas/ner}/external"
D=renascor-atlas-data/ner
mkdir -p "$DEST/datasets" "$DEST/models"
nice -n 19 rsync -a "$REMOTE:$D/external/SOURCES.tsv" "$REMOTE:$D/external/CONVERTED.tsv" "$DEST/"
for d in $(ssh "$REMOTE" "cd $D/external && ls -d */ | tr -d /"); do
  [ "$d" = aggregate ] && continue
  mkdir -p "$DEST/datasets/$d"
  nice -n 19 rsync -a --include='SOURCES.tsv' --include='counts.json' --include='pile_types_top.json' --include='*.md' --exclude='*' "$REMOTE:$D/external/$d/" "$DEST/datasets/$d/" || true
done
nice -n 19 rsync -a "$REMOTE:$D/external/aggregate/" --include='*/' --include='MANIFEST.tsv' --exclude='*' "$DEST/aggregate/" 2>/dev/null || true
nice -n 19 rsync -a "$REMOTE:$D/MODELS*.tsv" "$REMOTE:$D/VERIFY.tsv" "$DEST/models/" || true
nice -n 19 rsync -a --exclude='__pycache__' "$REMOTE:$D/model_code/" "$DEST/models/model_code/" || true
echo "pulled to $DEST"
