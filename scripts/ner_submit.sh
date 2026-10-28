#!/usr/bin/env bash
# Run on Hypatia (the login node only submits): scripts/ner_submit.sh slug1 slug2 ...   or   scripts/ner_submit.sh all
# Env: CONCURRENT (array tasks at once, default 3: 3 x 8 cpus = 24 cpus), LIMIT (first N files, for tests), NER_VERSION (default data_v1)
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: $0 slug... | all" >&2; exit 2; }
cd "$HOME/renascor-atlas"
if [ "$1" = "all" ]; then
  SLUGS=$(ls src/ratlas/collections/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -v -e '^__init__$' -e '^generic_tei$')
else
  SLUGS="$*"
fi
SLUGS=$(echo $SLUGS)      # one line, single spaces
N=$(echo $SLUGS | wc -w)
mkdir -p "$HOME/renascor-atlas-data/logs"
SLUGS="$SLUGS" sbatch --parsable --export=ALL,SLUGS="$SLUGS",LIMIT="${LIMIT:-}",NER_VERSION="${NER_VERSION:-data_v1}" \
  --output="$HOME/renascor-atlas-data/logs/%x-%A_%a.out" --array="0-$((N-1))%${CONCURRENT:-3}" slurm/ner_extract.sbatch
