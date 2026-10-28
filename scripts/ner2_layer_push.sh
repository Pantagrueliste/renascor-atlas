#!/usr/bin/env bash
# Ship ONLY the files of the NER layer (stage 1-3) to Hypatia; nothing else of the repository is touched.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ssh hypatia 'mkdir -p ~/renascor-atlas/src/ratlas/ner ~/renascor-atlas/slurm ~/renascor-atlas/scripts ~/renascor-atlas/tests ~/renascor-atlas-data/ner/layer_v1/logs'
FILES="src/ratlas/ner/layerpred.py $(ls scripts/ner2_layer_* slurm/ner2_layer_* tests/test_ner2_layer* 2>/dev/null | tr '\n' ' ')"
for f in $FILES; do scp -q "$f" "hypatia:renascor-atlas/$f"; done
echo "pushed: $FILES"
