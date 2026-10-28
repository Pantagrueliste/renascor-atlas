#!/bin/bash
# CPU smoke test of stage 2 (all four models, a handful of documents): checks the whole path including the agreement and the parquet files.
set -u
D="$HOME/renascor-atlas-data/ner"; L="$D/layer_v1"
cd "$HOME/renascor-atlas"
rm -rf "$L/smoke"; mkdir -p "$L/smoke"
SH=$(ls "$L/extract/texts/${SMOKE_LANG:-it}"/ | head -1)
python scripts/ner2_layer_run.py --texts-root "$L/extract" --shards "texts/${SMOKE_LANG:-it}/$SH" --docs "$L/extract/docs_read.parquet" --out "$L/smoke" --models-dir "$D/models" \
  --limit-docs "${SMOKE_DOCS:-8}" --chunk-tokens 60000 --pack-chars 300000 --workers 2
python - <<PY
import pyarrow.parquet as pq, glob
for f in sorted(glob.glob("$L/smoke/spans/*.parquet")):
    t = pq.read_table(f).to_pandas()
    print(f.split('/')[-1], len(t), 'spans')
    print(t.drop(columns=['doc_id']).head(8).to_string())
PY
