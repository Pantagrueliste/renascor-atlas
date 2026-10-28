#!/bin/bash
# NER v2: assemble test_v2.jsonl (all 7 editions) and test_v2_loc.jsonl (loc_valid passages of the 6 editions with LOC) from the repaired passages. Via slurm/ner2_cpu.sbatch.
set -u
V1="$ND/data_v1"; O="$ND/v2/test2"
$PY -I -m ratlas.ner.ner2_testset assemble --v1-test "$V1/test.jsonl" --passages "$O/passages" --out "$O/test_v2.jsonl" --loc-out "$O/test_v2_loc.jsonl" --stats "$O/test_v2_stats.json" > "$O/assemble.log"
tail -c 2500 "$O/assemble.log"
