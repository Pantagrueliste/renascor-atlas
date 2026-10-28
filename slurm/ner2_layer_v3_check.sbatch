#!/bin/bash
# NER layer v3: duplicate control of tei / seen / extra (against each other and against the layer) and the per-form facts of the corrections (ctx, p_per).
#   sbatch --output=$HOME/renascor-atlas-data/ner/layer_v3/logs/%x-%j.out slurm/ner2_layer_v3_check.sbatch [dedup|fix|both]
#SBATCH --job-name=ner_v3_check
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -uo pipefail
PART="${1:-both}"
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
echo "[v3_check $PART] $(date +%T) on $(hostname)"
nice -n 10 .venv/bin/python -I scripts/ner2_layer_v3_check.py --data "$D" --out "$D/layer_v3" --procs 16 --part "$PART"
rc=$?; echo "[v3_check] done rc=$rc $(date +%T)"; exit $rc
