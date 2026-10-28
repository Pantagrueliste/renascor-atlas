#!/usr/bin/env bash
# Ship ONLY the NER files to Hypatia (no --delete, nothing else of the repository is touched), and the small meta tables.
#   scripts/ner_push.sh
# One ssh connection (ControlMaster) for all the transfers: a login takes several seconds on the cluster.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${RATLAS_REMOTE:-hypatia}"
META="${NER_META:-/Users/clem/Projects/Papers/RQ/derived/atlas/ner/meta}"
SSH="ssh -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=$HOME/.ssh/cm-ner-%C"
$SSH "$REMOTE" 'mkdir -p ~/renascor-atlas/src/ratlas/ner ~/renascor-atlas/slurm ~/renascor-atlas/scripts ~/renascor-atlas/tests ~/renascor-atlas-data/ner/meta'
cd "$REPO"
nice -n 19 rsync -a -e "$SSH" --exclude __pycache__ --exclude '*.pyc' src/ratlas/ner/ "$REMOTE:renascor-atlas/src/ratlas/ner/"
FILES=$(ls slurm/ner_* scripts/ner_* tests/test_ner_*.py 2>/dev/null)
nice -n 19 rsync -a -e "$SSH" --relative $FILES "$REMOTE:renascor-atlas/"
[ -d "$META" ] && nice -n 19 rsync -a -e "$SSH" "$META/" "$REMOTE:renascor-atlas-data/ner/meta/"
$SSH -O exit "$REMOTE" 2>/dev/null || true
echo "pushed the NER files to $REMOTE"
