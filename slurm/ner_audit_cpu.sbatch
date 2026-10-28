#!/bin/bash
# Independent audit, CPU job on the debug partition (<= 24 cpus, 2 h): SCRIPT (path relative to ~/renascor-atlas) and ARGS in the environment.
#   sbatch --job-name=ner-audit-leak --export=ALL,SCRIPT=scripts/ner_audit_leak.py,ARGS="--data ... --agg ... --out ..." slurm/ner_audit_cpu.sbatch
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
source "$D/venv_train/bin/activate"
export PYTHONPATH="$HOME/renascor-atlas/src" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
cd "$HOME/renascor-atlas"
echo "[ner-audit-cpu] $(date +%T) $(hostname) $SCRIPT"
nice -n 10 python "$SCRIPT" $ARGS
rc=$?
echo "[ner-audit-cpu] exit $rc $(date +%T)"
exit $rc
