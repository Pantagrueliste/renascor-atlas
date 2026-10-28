#!/usr/bin/env bash
# Copy the reports and tables (no text, no passages) of a data version from Hypatia to the RQ derived folder.
#   scripts/ner_pull.sh [data_v1]
set -euo pipefail
V="${1:-data_v1}"
REMOTE="${RATLAS_REMOTE:-hypatia}"
DEST="${NER_DEST:-/Users/clem/Projects/Papers/RQ/derived/atlas/ner}"
mkdir -p "$DEST/$V"
nice -n 19 rsync -a "$REMOTE:renascor-atlas-data/ner/$V/DATASHEET.md" "$REMOTE:renascor-atlas-data/ner/$V/labels.json" "$REMOTE:renascor-atlas-data/ner/$V/dataset_info.json" "$DEST/$V/"
nice -n 19 rsync -a "$REMOTE:renascor-atlas-data/ner/$V/tables/" "$DEST/$V/tables/"
nice -n 19 rsync -a "$REMOTE:renascor-atlas-data/ner/$V/completeness/" "$DEST/$V/completeness/"
nice -n 19 rsync -a "$REMOTE:renascor-atlas-data/ner/$V/raw/summary/" "$DEST/$V/extraction_summary/"
cp "$DEST/$V/DATASHEET.md" "$DEST/DATASHEET.md"
echo "copied to $DEST/$V (reports only)"
