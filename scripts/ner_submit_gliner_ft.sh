#!/usr/bin/env bash
# Run on Hypatia. scripts/ner_submit_gliner_ft.sh NAME PARTITION "gliner_ft arguments" [AFTER_JOBID]
set -euo pipefail
NAME="$1"; PART="$2"; ARGS="$3"; AFTER="${4:-}"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency="afterany:$AFTER")
export ARGS="--name $NAME $ARGS $SET_ARGS"
sbatch --parsable --partition="$PART" --job-name="ner-$NAME" "${DEP[@]}" --export=ALL slurm/ner_gliner_ft.sbatch
