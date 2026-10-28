"""NER layer v1, stage 3 (CPU): assemble spans.parquet, docs.parquet and the summary tables.

    python scripts/ner2_layer_merge.py --layer ~/renascor-atlas-data/ner/layer_v1

Reads ``extract/docs_all.parquet`` (inventory), ``extract/docs_read.parquet`` (documents read) and ``spans/*.parquet`` (one file per chunk).
Writes under --layer:
  spans.parquet        every span of the main model: doc_id, passage, w_start, w_end, char_start, char_end, surface, conf, p_per, n_agree, n_overlap, lang, year, collection
  docs.parquet         one row per document read (status ok): collection, lang, year, date_confidence, tier, licence_tag, n_words, n_tok, n_dropcap, run, n_spans, ...
  summary/*.tsv        inventory (words by collection / language / status), counts of the layer by language, collection, decade, confidence
A span is "recommended" when conf >= 0.98 and n_agree >= 2 (of the four top models).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    a = ap.parse_args()
    L = Path(a.layer)
    S = L / 'summary'
    S.mkdir(exist_ok=True)
    files = sorted((L / 'spans').glob('*.parquet'))
    print(len(files), 'chunk files')
    # spans.parquet: written chunk by chunk (it can be large)
    writer = None
    parts = []
    tot = 0
    for f in files:
        t = pq.read_table(f)
        if writer is None:
            writer = pq.ParquetWriter(L / 'spans.parquet', t.schema, compression='zstd')
        writer.write_table(t.cast(writer.schema))
        tot += t.num_rows
        d = t.select(['doc_id', 'conf', 'n_agree', 'lang']).to_pandas()
        d['hi'] = d['conf'] >= 0.9
        d['c98'] = d['conf'] >= 0.98
        d['rec'] = d['c98'] & (d['n_agree'].fillna(0) >= 2)
        g = d.groupby('doc_id').agg(n_spans=('conf', 'size'), n_spans_c90=('hi', 'sum'), n_spans_c98=('c98', 'sum'), n_recommended=('rec', 'sum')).reset_index()
        parts.append(g)
    if writer is not None:
        writer.close()
    print('spans', tot)
    sp = pd.concat(parts, ignore_index=True).groupby('doc_id', as_index=False).sum() if parts else pd.DataFrame(columns=['doc_id', 'n_spans'])
    # documents: those with text read; run = a chunk file exists for the document (spans or not): known from the log of done chunks
    dr = pd.read_parquet(L / 'extract' / 'docs_read.parquet')
    ok = dr[dr['status'] == 'ok'].copy()
    done = set()
    for f in sorted((L / 'chunks').glob('*.docs')):
        done.update(f.read_text().split())
    ok = ok.merge(sp, on='doc_id', how='left')
    for c in ('n_spans', 'n_spans_c90', 'n_spans_c98', 'n_recommended'):
        ok[c] = ok[c].fillna(0).astype(int)
    ok['run'] = ok['doc_id'].isin(done) if done else ok['n_spans'] > 0
    ok.to_parquet(L / 'docs.parquet', index=False)
    print('docs', len(ok))
    # inventory
    inv = pd.read_parquet(L / 'extract' / 'docs_all.parquet')
    inv = inv[(inv['status'] != 'year_out')]
    inv['words'] = np.where(inv['n_words'] > 0, inv['n_words'], inv['db_words'].fillna(0)).astype(np.int64)
    inv['lang_db'] = inv['lang_db'].fillna('')
    inv.groupby(['collection', 'lang_db', 'status']).agg(docs=('doc_id', 'size'), words=('words', 'sum')).reset_index().to_csv(S / 'inventory_collection_lang_status.tsv', sep='\t', index=False)
    okk = inv[inv['status'] == 'ok']
    okk.groupby(['collection', 'lang']).agg(docs=('doc_id', 'size'), words=('n_words', 'sum'), tokens=('n_tok', 'sum')).reset_index().to_csv(
        S / 'inventory_ok_collection_lang.tsv', sep='\t', index=False)
    inv.groupby(['status']).agg(docs=('doc_id', 'size'), words=('words', 'sum')).reset_index().to_csv(S / 'inventory_by_status.tsv', sep='\t', index=False)
    # the layer
    if tot:
        sp_t = pq.read_table(L / 'spans.parquet', columns=['lang', 'conf', 'n_agree', 'year', 'collection'])
        d = sp_t.to_pandas()
        d['c98'] = d['conf'] >= 0.98
        d['rec'] = d['c98'] & (d['n_agree'].fillna(0) >= 2)
        d['decade'] = (d['year'] // 10 * 10)
        d.groupby('lang').agg(spans=('conf', 'size'), c90=('conf', lambda x: int((x >= 0.9).sum())), c98=('c98', 'sum'), recommended=('rec', 'sum')).reset_index().to_csv(
            S / 'layer_by_lang.tsv', sep='\t', index=False)
        d.groupby(['collection', 'lang']).agg(spans=('conf', 'size'), c98=('c98', 'sum'), recommended=('rec', 'sum')).reset_index().to_csv(S / 'layer_by_collection.tsv', sep='\t', index=False)
        d.groupby(['lang', 'decade']).agg(spans=('conf', 'size'), recommended=('rec', 'sum')).reset_index().to_csv(S / 'layer_by_lang_decade.tsv', sep='\t', index=False)
    # throughput of the GPU jobs, from their logs
    import re
    rows = []
    gpu_of = {}
    for lg in sorted((L / 'logs').glob('ner_layer_*.out')):
        job = lg.stem.rsplit('-', 1)[-1]
        txt = lg.read_text(errors='replace')
        m = re.search(r'gpu: ([^\n]+?),', txt)
        gpu_of[job] = m.group(1) if m else ''
        for mm in re.finditer(r'(\S+\.parquet): (\d+) docs, (\d+) tokens, (\d+) spans \((\d+) >= 0.98\), (\d+) passages re-run; main (\d+)s \((\d+) tok/s\), others (\d+)s, total (\d+)s', txt):
            rows.append((job, gpu_of[job]) + mm.groups())
    thr = pd.DataFrame(rows, columns=['job', 'gpu', 'chunk', 'docs', 'tokens', 'spans', 'spans_c98', 'passages_rerun', 'main_s', 'main_tok_s', 'others_s', 'total_s'])
    thr = thr.sort_values('job').drop_duplicates('chunk', keep='last')        # a chunk re-run by a later job replaces the earlier one
    thr = thr[thr['chunk'].isin({f.name for f in files})]
    thr.to_csv(S / 'chunks_throughput.tsv', sep='\t', index=False)
    run = ok[ok['run']]
    run.groupby('lang').agg(docs=('doc_id', 'size'), words=('n_words', 'sum'), tokens=('n_tok', 'sum'), spans=('n_spans', 'sum'), c98=('n_spans_c98', 'sum'),
                            recommended=('n_recommended', 'sum')).reset_index().to_csv(S / 'run_by_lang.tsv', sep='\t', index=False)
    print((S / 'run_by_lang.tsv').read_text())


if __name__ == '__main__':
    main()
