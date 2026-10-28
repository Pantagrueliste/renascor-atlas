#!/bin/sh
# Assembles on the Mac the tables of the v1 and Spanish runs that ner2_layer_v3_build.py reads as --work, from Hypatia (about 22 MB).
#   sh scripts/v29_layer_workdir.sh WORKDIR
# Then, with the layer v3 folder (RQ/derived/atlas/ner/v2/layer/v3) and the globe rows:
#   uv run python scripts/ner2_layer_v3_build.py --work WORKDIR --v3 RQ/derived/atlas/ner/v2/layer/v3 --globe RQ/derived/atlas/v2.4/globe_places.json \
#       --out-dir OUT --titles [--drop-ecclesiastical]
# and copy OUT/model_layer_v2.json and model_layer_v2_docs.json to RQ/derived/atlas/page_kit/v29/layer/ (the default model folder of build_v29.py).
set -e
W="${1:?usage: v29_layer_workdir.sh WORKDIR}"
H=hypatia:renascor-atlas-data/ner/layer_v1
mkdir -p "$W/es" "$W/fix"
scp -q "$H/geo/points/forms_placed.parquet" "$W/forms_placed.parquet"
scp -q "$H/geo/doc_forms.parquet" "$W/doc_forms.parquet"
scp -q "$H/docs.parquet" "$W/docs_v1.parquet"
scp -q "$H/es_mdeberta/geo/points/forms_placed.parquet" "$W/es/forms_placed.parquet"
scp -q "$H/es_mdeberta/geo/doc_forms.parquet" "$W/es/doc_forms.parquet"
scp -q "$H/es_mdeberta/docs.parquet" "$W/es/docs_es.parquet"
scp -q "$H/fix/ctx.parquet" "$H/fix/form_pper.parquet" "$H/fix/titles.json" "$W/fix/"
echo "work folder ready: $W"
