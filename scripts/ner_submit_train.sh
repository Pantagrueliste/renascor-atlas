#!/usr/bin/env bash
# Run on Hypatia (the login node only submits). Submit one training run.
#   scripts/ner_submit_train.sh NAME PARTITION "train.py arguments without --name/--out" [AFTER_JOBID]
# AFTER_JOBID: wait for that job to end (lanes of at most two runs at a time).
set -euo pipefail
NAME="$1"; PART="$2"; ARGS="$3"; AFTER="${4:-}"
cd "$HOME/renascor-atlas"
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency="afterany:$AFTER")
export ARGS="--name $NAME $ARGS"   # through the environment: the values contain commas and ';'
sbatch --parsable --partition="$PART" --job-name="ner-$NAME" "${DEP[@]}" --export=ALL slurm/ner_train.sbatch
