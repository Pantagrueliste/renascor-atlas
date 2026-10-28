#!/bin/bash
# NER layer extra, CPU job on the debug partition (<= 24 cpus, 2 h): SCRIPT (relative to ~/renascor-atlas) and ARGS in the environment; VENV = repo .venv (lxml) or train.
#   sbatch --job-name=ner2_layer_extra_x --export=ALL,SCRIPT=scripts/ner2_layer_extract.py,ARGS="run ..." slurm/ner2_layer_cpu.sbatch
#SBATCH --partition=debug
#SBATCH --cpus-per-task=22
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_extra/logs/%x-%j.out
set -uo pipefail
set -f
D="$HOME/renascor-atlas-data/ner"
if [ "${VENV:-repo}" = "train" ]; then source "$D/venv_train/bin/activate"; PY=python; else PY="$HOME/renascor-atlas/.venv/bin/python"; fi
export PYTHONPATH="$HOME/renascor-atlas/src" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 HF_HUB_CACHE="$D/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
cd "$HOME/renascor-atlas"
echo "[ner2-layer-cpu] $(date +%T) $(hostname) $SCRIPT"
if [[ "$SCRIPT" == *.sh ]]; then nice -n 10 bash "$SCRIPT" ${ARGS:-}; else nice -n 10 $PY "$SCRIPT" ${ARGS:-}; fi
rc=$?
echo "[ner2-layer-cpu] exit $rc $(date +%T)"
exit $rc
