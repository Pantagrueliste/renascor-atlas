#!/bin/bash
# Candidate model weights -> ~/renascor-atlas-data/ner/hf_cache (public Hugging Face repos, no token, no upload). Network only, 2 cpus.
#   sbatch slurm/ner_external_models.sbatch                       all
#   sbatch --export=ALL,ROLE=llm slurm/ner_external_models.sbatch   one role (encoder | offshelf | hist | latin | llm)
#SBATCH --job-name=ratlas-ner-ext-models
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:50:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/logs/%x-%j.out
set -euo pipefail
D="$HOME/renascor-atlas-data/ner"
cd "$HOME/renascor-atlas"
export HF_HOME="$D/hf_cache/home" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
echo "[ner ext models] $(date +%T) on $(hostname) role=${ROLE:-all}"
nice -n 10 "$D/venv_hf/bin/python" -I scripts/ner_external_models.py --cache "$D/hf_cache" ${ROLE:+--role "$ROLE"}
echo "[ner ext models] done $(date +%T)"
