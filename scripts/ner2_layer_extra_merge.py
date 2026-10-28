"""NER layer, extra documents, after the GPU job: assemble spans.parquet / docs.parquet like layer_v1 and write the report tables.

    python scripts/ner2_layer_extra_merge.py --layer ~/renascor-atlas-data/ner/layer_extra

Steps: ner2_layer_merge (main run: spans.parquet, docs.parquet, summary/), ner2_layer_es_merge (Spanish run under es_mdeberta/), then the Spanish rows of
docs.parquet are filled from the Spanish run (run = True when a chunk file lists the document). Writes summary/extra_by_collection_lang.tsv (what happened to every
document of the run, by collection and language) and summary/extra_ok_by_collection_lang.tsv (documents and words that went to the model).
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def call(script: str, argv: list[str]):
    old = sys.argv
    sys.argv = [script] + argv
    try:
        runpy.run_path(str(HERE / script), run_name='__main__')
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=Path, required=True)
    a = ap.parse_args()
    L = a.layer.expanduser()
    call('ner2_layer_merge.py', ['--layer', str(L)])
    es = L / 'es_mdeberta'
    if (es / 'spans').exists() and any((es / 'spans').glob('*.parquet')):
        call('ner2_layer_es_merge.py', ['--layer', str(L), '--es', str(es)])
        d = pd.read_parquet(L / 'docs.parquet')
        e = pd.read_parquet(es / 'docs.parquet').set_index('doc_id')
        done = set()
        for f in sorted((es / 'chunks').glob('*.docs')):
            done.update(f.read_text().split())
        m = d['lang'] == 'es'
        for c in ('n_spans', 'n_spans_c98', 'n_recommended'):
            d.loc[m, c] = d.loc[m, 'doc_id'].map(e[c]).fillna(0).astype(int).values
        d.loc[m, 'run'] = d.loc[m, 'doc_id'].isin(done).values
        d.to_parquet(L / 'docs.parquet', index=False)
    S = L / 'summary'
    S.mkdir(exist_ok=True)
    allr = pd.read_parquet(L / 'extract' / 'docs_all.parquet')
    allr['words'] = allr['n_words'].where(allr['n_words'] > 0, allr['db_words'].fillna(0)).astype('int64')
    allr.groupby(['collection', 'lang_db', 'status']).agg(docs=('doc_id', 'size'), words=('words', 'sum')).reset_index().to_csv(
        S / 'extra_by_collection_lang.tsv', sep='\t', index=False)
    d = pd.read_parquet(L / 'docs.parquet')
    d.groupby(['collection', 'lang']).agg(docs=('doc_id', 'size'), run=('run', 'sum'), words=('n_words', 'sum'), spans=('n_spans', 'sum'),
                                          recommended=('n_recommended', 'sum')).reset_index().to_csv(S / 'extra_ok_by_collection_lang.tsv', sep='\t', index=False)
    print(d.groupby('lang').agg(docs=('doc_id', 'size'), run=('run', 'sum'), words=('n_words', 'sum'), spans=('n_spans', 'sum'), recommended=('n_recommended', 'sum')))


if __name__ == '__main__':
    main()
