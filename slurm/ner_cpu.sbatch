#!/bin/bash
# CPU job on the debug partition (<= 24 cpus, 2 h): MODULE (python module) and ARGS in the environment.
#   sbatch --job-name=ner-gaz --export=ALL,MODULE=ratlas.ner.gazetteer,ARGS="--name gazetteer_ours ..." slurm/ner_cpu.sbatch
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f   # ARGS holds glob patterns for the Python side: the shell must not expand them
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" PYTHONUNBUFFERED=1
echo "[ner-cpu] $(date +%T) $(hostname) $MODULE"
nice -n 10 python -m "$MODULE" $ARGS --out "$D/runs"
rc=$?
echo "[ner-cpu] exit $rc $(date +%T)"
exit $rc
