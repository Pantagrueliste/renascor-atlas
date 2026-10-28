#!/usr/bin/env bash
# Copy the metrics (no predictions, no weights, no text) of all NER runs from Hypatia to the RQ derived folder.
#   scripts/ner_pull_runs.sh
set -euo pipefail
REMOTE="${RATLAS_REMOTE:-hypatia}"
DEST="${NER_DEST:-/Users/clem/Projects/Papers/RQ/derived/atlas/ner}"
mkdir -p "$DEST/runs"
nice -n 19 rsync -a --include='*/' --include='metrics*.json' --include='config.json' --include='done.json' --exclude='*' "$REMOTE:renascor-atlas-data/ner/runs/" "$DEST/runs/"
echo "copied the metrics of $(ls "$DEST/runs" | wc -l) runs to $DEST/runs"
