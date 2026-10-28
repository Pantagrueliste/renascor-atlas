#!/bin/bash
# NER layer, TEI editions, step 3 only (tables, chunks, editor tags), CPU, debug partition.
#SBATCH --job-name=layer_tei_asm
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/cag30/renascor-atlas-data/ner/layer_tei/logs/assemble-%j.out
set -uo pipefail
T="$HOME/renascor-atlas-data/ner/layer_tei"
cd "$HOME/renascor-atlas"
$HOME/renascor-atlas/.venv/bin/python -I scripts/ner2_tei_assemble.py --work "$T/work" --out "$T"
