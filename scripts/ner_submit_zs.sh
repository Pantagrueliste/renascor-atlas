#!/usr/bin/env bash
# Run on Hypatia. Submit an off-the-shelf evaluation.
#   scripts/ner_submit_zs.sh NAME BACKEND VENV PARTITION "zeroshot.py arguments (--model, --map, --labels ...)" [AFTER_JOBID]
# The sets are those of scripts/ner_sets.sh. The variable SET_ARGS_OVERRIDE replaces them to run on other sets.
set -euo pipefail
NAME="$1"; BACKEND="$2"; VENV="$3"; PART="$4"; ARGS="$5"; AFTER="${6:-}"
cd "$HOME/renascor-atlas"
source scripts/ner_sets.sh
DEP=(); [ -n "$AFTER" ] && DEP=(--dependency="afterany:$AFTER")
export BACKEND VENV ARGS="--name $NAME $ARGS ${SET_ARGS_OVERRIDE:-$SET_ARGS}"   # through the environment: the values contain commas
sbatch --parsable --partition="$PART" --job-name="ner-zs-$NAME" "${DEP[@]}" --export=ALL slurm/ner_zeroshot.sbatch
