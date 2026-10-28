#!/bin/bash
# NER v2: candidate groups from the predictions of all systems (CPU, via slurm/ner2_cpu.sbatch). Writes $ND/v2/test2/cand/cand_<edition>.jsonl and cand_summary.json.
set -u
O="$ND/v2/test2"
$PY -I -m ratlas.ner.ner2_cand --data "$O/test_v2_loc.jsonl" --pred-dir "$O/pred" --out-dir "$O/cand" ${EXCLUDE:+--exclude "$EXCLUDE"}
ls -la "$O/cand"
