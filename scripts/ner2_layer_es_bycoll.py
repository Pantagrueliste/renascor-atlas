"""Spanish rerun: placed mentions and documents by collection (file ``placed_by_collection.tsv`` for ner2_layer_geo_tables.py).

    python scripts/ner2_layer_es_bycoll.py --es ES_DIR --geo-sub geo
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--es', type=Path, required=True)
ap.add_argument('--geo-sub', default='geo')
a = ap.parse_args()
G = a.es / a.geo_sub
F = pd.read_parquet(G / 'points/forms_placed.parquet', columns=['lang', 'key', 'fate'])
F = F[F['fate'] == 'placed']
d = pd.read_parquet(G / 'doc_forms.parquet').merge(F[['lang', 'key']], on=['lang', 'key'])
c = pd.read_parquet(a.es / 'docs.parquet', columns=['doc_id', 'collection'])
d = d.merge(c, on='doc_id')
out = d.groupby('collection').agg(docs=('doc_id', 'nunique'), mentions=('n', 'sum')).reset_index().sort_values('mentions', ascending=False)
out.to_csv(G / 'points/placed_by_collection.tsv', sep='\t', index=False)
print(out.to_string(index=False))
