#!/bin/bash
# Stage 1 on Hypatia (CPU job): verify the archive copies, extract, merge.
set -u
L="$HOME/renascor-atlas-data/ner/layer_v1"
cd "$HOME/renascor-atlas"
PY="$HOME/renascor-atlas/.venv/bin/python"
echo "=== sha256 of the archive tarballs"
( cd "$L/archive_in" && grep -F -f <(ls *.tar.gz) SHA256SUMS | sha256sum -c - 2>&1 | grep -v ': OK$' ; echo "checked $(ls *.tar.gz | wc -l) tarballs" )
echo "=== extract"
$PY -I scripts/ner2_layer_extract.py run --archive "$L/archive_in" --db "$L/meta_in/corpus_master_FROZEN_2026-06-26.sqlite" --meta "$L/meta_in" --out "${OUTDIR:-$L/extract}" --procs "${PROCS:-22}"
echo "=== merge"
$PY -I scripts/ner2_layer_extract.py merge --archive "$L/archive_in" --db "$L/meta_in/corpus_master_FROZEN_2026-06-26.sqlite" --out "${OUTDIR:-$L/extract}"
echo "=== plan (English ordered by decade) and work list"
$PY -I scripts/ner2_layer_plan.py --extract "${OUTDIR:-$L/extract}" --procs "${PROCS:-22}"
