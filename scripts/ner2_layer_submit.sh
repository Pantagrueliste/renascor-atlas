#!/bin/bash
# Submit ONE GPU job of the layer run, respecting the shared GPU rule: the account may run at most two NER GPU jobs (names starting with "ner"); mine is
# queued or running alone. Run on the Hypatia login node (only submits). Usage: scripts/ner2_layer_submit.sh JOBNAME [EXTRA_SBATCH_ARGS...]
set -u
NAME="$1"; shift
cd "$HOME/renascor-atlas"
mine=$(squeue -u cag30 -h -o "%j %P" | awk -v n="ner_layer" 'index($1,n)==1 && $2 ~ /^gpu/' | wc -l)
if [ "$mine" -ge 1 ]; then echo "one of my GPU jobs is already queued or running"; exit 2; fi
ner=$(squeue -u cag30 -h -t R,PD -o "%j %P" | awk 'index($1,"ner")==1 && $2 ~ /^gpu/' | wc -l)
if [ "$ner" -ge 2 ]; then echo "two NER GPU jobs already running or pending ($ner): wait"; exit 3; fi
sbatch --job-name="$NAME" --export=ALL,SHARDS_FILE="$HOME/renascor-atlas-data/ner/layer_v1/extract/plan/worklist.txt" "$@" slurm/ner2_layer_gpu.sbatch
