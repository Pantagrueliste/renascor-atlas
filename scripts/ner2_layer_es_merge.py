"""Spanish rerun: assemble es_mdeberta/spans.parquet (same schema as layer_v1/spans.parquet), a docs.parquet and the links the geocoding steps need.

    python scripts/ner2_layer_es_merge.py --layer ~/renascor-atlas-data/ner/layer_v1 --es ~/renascor-atlas-data/ner/layer_v1/es_mdeberta

Writes in --es: spans.parquet (all chunk files of ``spans/``), docs.parquet (the Spanish rows of layer_v1/docs.parquet, counts recomputed), extract/texts/es (symlink).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=Path, required=True)
    ap.add_argument('--es', type=Path, required=True)
    ap.add_argument('--min-conf', type=float, default=0.5)
    ap.add_argument('--min-agree', type=int, default=2)
    a = ap.parse_args()
    files = sorted((a.es / 'spans').glob('*.parquet'))
    tabs = [pq.read_table(f) for f in files]
    schema = tabs[0].schema
    w = pq.ParquetWriter(a.es / 'spans.parquet', schema, compression='zstd')
    for t in tabs:
        w.write_table(t.cast(schema))
    w.close()
    s = pd.read_parquet(a.es / 'spans.parquet', columns=['doc_id', 'conf', 'n_agree'])
    s['rec'] = (s['conf'] >= a.min_conf) & (s['n_agree'].fillna(0) >= a.min_agree)
    s['c98'] = s['conf'] >= 0.98
    g = s.groupby('doc_id').agg(n_spans=('conf', 'size'), n_spans_c98=('c98', 'sum'), n_recommended=('rec', 'sum')).reset_index()
    d = pd.read_parquet(a.layer / 'docs.parquet')
    d = d[d['lang'] == 'es'].drop(columns=['n_spans', 'n_spans_c98', 'n_recommended']).merge(g, on='doc_id', how='left')
    for c in ('n_spans', 'n_spans_c98', 'n_recommended'):
        d[c] = d[c].fillna(0).astype(int)
    d.to_parquet(a.es / 'docs.parquet', index=False)
    tx = a.es / 'extract' / 'texts'
    tx.mkdir(parents=True, exist_ok=True)
    if not (tx / 'es').exists():
        os.symlink(a.layer / 'extract' / 'texts' / 'es', tx / 'es')
    print(len(files), 'chunks', len(s), 'spans', int(s['rec'].sum()), 'recommended', int(s['c98'].sum()), 'conf>=0.98', len(d), 'docs')


if __name__ == '__main__':
    main()
