#!/bin/bash
# NER v2: rebuild the 7 held-out test editions of data_v1 with the layout repairs (tei.Options.repair_ws, dehyph) at the v1 passage regions, then assemble test_v2.
# Control first (no repair: must reproduce the v1 texts exactly), then the repaired run. Run through slurm/ner2_cpu.sbatch.
set -u
V1="$ND/data_v1"; O="$ND/v2/test2"; mkdir -p "$O/control" "$O/passages"
W="${SLURM_CPUS_PER_TASK:-4}"
EDS="elan_crenum elan_fnvr hainhofer_reisen kaiserin_eleonora_magdalena koenigsfelden reed sozinianische_briefwechsel"
for E in $EDS; do
  echo "=== control $E"
  $PY -I -m ratlas.ner.ner2_testset rebuild "$E" --raw "$V1/raw/passages/$E.jsonl.gz" --out "$O/control/$E.jsonl.gz" --meta "$ND/meta" --workers "$W"
done
for E in $EDS; do
  echo "=== repair $E"
  $PY -I -m ratlas.ner.ner2_testset rebuild "$E" --raw "$V1/raw/passages/$E.jsonl.gz" --out "$O/passages/$E.jsonl.gz" --meta "$ND/meta" --workers "$W" --repair-ws --dehyph
done
echo "=== assemble"
$PY -I -m ratlas.ner.ner2_testset assemble --v1-test "$V1/test.jsonl" --passages "$O/passages" --out "$O/test_v2.jsonl" --stats "$O/test_v2_stats.json" > "$O/assemble.log"
tail -c 3000 "$O/assemble.log"
