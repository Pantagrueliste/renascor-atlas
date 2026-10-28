"""NER layer, TEI editions, step 3 (CPU, Hypatia): final tables, texts in the format of the layer run, editor tags.

    python scripts/ner2_tei_assemble.py --work layer_tei/work --out layer_tei

Reads the output of ner2_tei_extract.py (docs_read.parquet, tags_read.parquet, texts/) and of ner2_tei_dedup.py (dedup/decisions.parquet) and writes

    docs.parquet            documents kept for the model (one row each, with the dedup and seen_in_training fields)
    chunks/<lang>/teiNNN.jsonl.gz   {"id", "t"} like extract/texts of the layer (input of ner2_layer_run.py); chunks/shards.tsv, chunks/worklist.txt
    editor_tags.parquet     doc_id, char_start, char_end, tag, type (+ ref, cert, nested, tag_kind) for the kept documents
    seen/                   the same three files for the documents the first pass excluded one by one (v1 train / dev / test, gold seed): not in the main set
    docs_all.parquet        every candidate with its final status and reason
    report_*.tsv            counts by collection and status

Final status: not_read / not_found / short / lang_other / no_body (nothing to run); dup_layer / dup_layer_exact / dup_set (duplicates, dropped); v1_doc / gold (side set,
``seen/``); keep. ``seen_in_training`` (kept documents) = some piece of the document is in the v1 lists, or >= SEEN_MIN of its sampled shingles are in the v1 train / dev /
test passages, the external aggregates or the gold seed.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEEN_MIN = 0.05
SHARD_TOKENS = 20_000_000
EX_COLS = ['ex_note_editorial', 'ex_note_other', 'ex_apparatus', 'ex_div_editorial', 'ex_head_editorial', 'ex_standoff', 'ex_diplomatic']
WORD = re.compile(r'\w+|[^\w\s]', re.U)


def tag_kind(tag, ty):
    ty = (ty or '').lower()
    if tag in ('placeName', 'settlement', 'country', 'region', 'geogName', 'geogFeat', 'bloc', 'district', 'pubPlace'):
        return 'place'
    if tag == 'persName':
        return 'person'
    if tag == 'orgName':
        return 'org'
    if tag in ('rs', 'name'):
        if re.match(r'^(place|location|loc|settlement|region|country|geog\w*|toponym|address|locus|regio|urbs|oppidum|insula|provincia|flumen|mons)$', ty):
            return 'place'
        if ty in ('person', 'perosn', 'peron', 'pers', 'persons'):
            return 'person'
        if ty in ('org', 'organisation', 'organization'):
            return 'org'
    return 'other'


def read_texts(work):
    texts = {}
    for p in sorted(glob.glob(str(Path(work) / 'texts' / '*.jsonl.gz'))):
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                texts[d['id']] = (d['t'], d.get('lang', ''))
    return texts


def write_set(out: Path, docs: pd.DataFrame, tags: pd.DataFrame, texts: dict, name_chunks='chunks'):
    out.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out / 'docs.parquet', index=False)
    tags.to_parquet(out / 'editor_tags.parquet', index=False)
    cdir = out / name_chunks
    shards = []
    for lang, g in docs.groupby('lang'):
        g = g.sort_values(['collection', 'doc_id'])
        (cdir / lang).mkdir(parents=True, exist_ok=True)
        cur, tok, k = [], 0, 0

        def flush():
            nonlocal cur, tok, k
            if not cur:
                return
            fn = f'{lang}/tei{k:03d}.jsonl.gz'
            with gzip.open(cdir / fn, 'wt', encoding='utf-8', compresslevel=3) as f:
                for did in cur:
                    f.write(json.dumps({'id': did, 't': texts[did][0]}, ensure_ascii=False) + '\n')
            sub = g[g['doc_id'].isin(cur)]
            shards.append((f'{name_chunks}/{fn}', lang, len(cur), int(sub['n_words'].sum()), int(sub['n_tok'].sum())))
            cur, tok, k = [], 0, k + 1

        for did, nt in zip(g['doc_id'], g['n_tok']):
            cur.append(did)
            tok += int(nt)
            if tok >= SHARD_TOKENS:
                flush()
        flush()
    with open(cdir / 'shards.tsv', 'w') as f:
        f.write('file\tlang\tn_docs\tn_words\tn_tok\n')
        for s in shards:
            f.write('\t'.join(map(str, s)) + '\n')
    with open(cdir / 'worklist.txt', 'w') as f:
        for s in shards:
            f.write(s[0] + '\n')
    return shards


def report(docs: pd.DataFrame, tags: pd.DataFrame, out: Path, label: str):
    g = docs.copy()
    kind = tags.groupby('doc_id')['tag_kind'].value_counts().unstack(fill_value=0) if len(tags) else pd.DataFrame()
    for k in ('place', 'person', 'org', 'other'):
        g[f'tags_{k}'] = g['doc_id'].map(kind[k]).fillna(0).astype(int) if k in kind.columns else 0
    g['ex_body'] = g[EX_COLS].sum(axis=1)
    g['ex_all'] = g['ex_body'] + g['ex_header'] + g['ex_front_back']
    agg = g.groupby('collection').agg(documents=('doc_id', 'size'), words=('n_words', 'sum'), tokens=('n_tok', 'sum'), editor_tags=('n_tags', 'sum'),
                                      tags_place=('tags_place', 'sum'), tags_person=('tags_person', 'sum'), tags_org=('tags_org', 'sum'), tags_other=('tags_other', 'sum'),
                                      docs_with_tags=('n_tags', lambda s: int((s > 0).sum())), kept_nows=('kept_nows', 'sum'), ex_body=('ex_body', 'sum'),
                                      ex_header=('ex_header', 'sum'), ex_front_back=('ex_front_back', 'sum'), ex_note_editorial=('ex_note_editorial', 'sum'),
                                      ex_note_other=('ex_note_other', 'sum'), ex_apparatus=('ex_apparatus', 'sum'), ex_div_editorial=('ex_div_editorial', 'sum'),
                                      ex_head_editorial=('ex_head_editorial', 'sum'), ex_standoff=('ex_standoff', 'sum'), ex_diplomatic=('ex_diplomatic', 'sum'),
                                      dropped_tags=('dropped_tags', 'sum'), seen_in_training=('seen_in_training', 'sum') if 'seen_in_training' in g else ('doc_id', 'size'),
                                      source_format=('source_format', 'first')).reset_index()
    agg['ex_editorial'] = agg['ex_body'] - agg['ex_diplomatic']
    agg['excl_share_editorial'] = (agg['ex_editorial'] / (agg['ex_body'] + agg['kept_nows']).replace(0, np.nan)).round(4)
    agg['excl_share_body'] = (agg['ex_body'] / (agg['ex_body'] + agg['kept_nows']).replace(0, np.nan)).round(4)
    agg['excl_share_file'] = ((agg['ex_body'] + agg['ex_header'] + agg['ex_front_back']) / (agg['ex_body'] + agg['ex_header'] + agg['ex_front_back'] + agg['kept_nows']).replace(0, np.nan)).round(4)
    tot = agg.drop(columns=['collection', 'source_format', 'excl_share_body', 'excl_share_file', 'excl_share_editorial']).sum()
    tot['collection'] = 'TOTAL'
    tot['source_format'] = ''
    tot['excl_share_editorial'] = round(tot['ex_editorial'] / max(tot['ex_body'] + tot['kept_nows'], 1), 4)
    tot['excl_share_body'] = round(tot['ex_body'] / max(tot['ex_body'] + tot['kept_nows'], 1), 4)
    tot['excl_share_file'] = round((tot['ex_body'] + tot['ex_header'] + tot['ex_front_back']) / max(tot['ex_body'] + tot['ex_header'] + tot['ex_front_back'] + tot['kept_nows'], 1), 4)
    agg = pd.concat([agg, pd.DataFrame([tot])], ignore_index=True)
    agg.to_csv(out / f'report_{label}_by_collection.tsv', sep='\t', index=False)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    docs = pd.read_parquet(work / 'docs_read.parquet')
    tags = pd.read_parquet(work / 'tags_read.parquet')
    dec = pd.read_parquet(work / 'dedup' / 'decisions.parquet')
    docs = docs.merge(dec[['doc_id', 'cover_layer', 'cover_set', 'seen_share', 'dedup', 'dedup_reason', 'dup_of', 'overlap_layer']], on='doc_id', how='left')

    def final(r):
        if r['status'] != 'ok':
            return r['status']
        if r['dedup'] == 'drop':
            return r['dedup_reason']
        if r['gold']:
            return 'gold'
        if r['v1_doc']:
            return 'v1_doc'
        return 'keep'
    docs['final'] = docs.apply(final, axis=1)
    docs['seen_share'] = docs['seen_share'].fillna(0.0)
    docs['seen_in_training'] = (docs['v1_hits'] > 0) | (docs['seen_share'] >= SEEN_MIN)
    docs['edition_in_v1'] = True
    docs['overlap_layer'] = docs['overlap_layer'].fillna(False).astype(bool)
    docs.to_parquet(out / 'docs_all.parquet', index=False)
    print(docs['final'].value_counts().to_dict())
    tags = tags.merge(docs[['doc_id', 'final']], on='doc_id', how='left')
    tags['tag_kind'] = [tag_kind(t, ty) for t, ty in zip(tags['tag'], tags['type'])]
    kc = tags.groupby('doc_id')['tag_kind'].value_counts().unstack(fill_value=0) if len(tags) else pd.DataFrame()
    for k, col in (('place', 'n_tags_place'), ('person', 'n_tags_pers'), ('org', 'n_tags_org'), ('other', 'n_tags_other')):
        docs[col] = docs['doc_id'].map(kc[k]).fillna(0).astype(int) if k in kc.columns else 0
    texts = read_texts(work)
    cols = ['doc_id', 'char_start', 'char_end', 'tag', 'type', 'ref', 'cert', 'nested', 'tag_kind']
    # validation of the ranges on the kept documents
    bad = 0
    for r in tags[tags['final'].isin(['keep', 'v1_doc', 'gold'])].itertuples():
        t = texts[r.doc_id][0]
        if not (0 <= r.char_start < r.char_end <= len(t)) or not t[r.char_start:r.char_end].strip():
            bad += 1
    print('editor tag ranges out of the text or blank:', bad)
    sets = {'main': docs['final'] == 'keep', 'seen': docs['final'].isin(['v1_doc', 'gold'])}
    summary = []
    for name, mask in sets.items():
        d = docs[mask].copy()
        tg = tags[tags['doc_id'].isin(set(d['doc_id']))][cols].copy()
        base = out if name == 'main' else out / 'seen'
        shards = write_set(base, d, tg, texts)
        rep = report(d, tg, out, name)
        print(name, len(d), 'documents', int(d['n_words'].sum()), 'words', len(tg), 'tags', len(shards), 'shards')
        summary.append((name, len(d), int(d['n_words'].sum()), len(tg)))
    tt = tags[tags['final'].isin(['keep', 'v1_doc', 'gold'])].copy()
    tt['collection'] = tt['doc_id'].str.split('/').str[0]
    tt['set'] = np.where(tt['final'] == 'keep', 'main', 'seen')
    tt.groupby(['collection', 'tag', 'type', 'tag_kind', 'set']).size().rename('n').reset_index().sort_values(['collection', 'tag', 'n'], ascending=[True, True, False])\
        .to_csv(out / 'report_tag_types.tsv', sep='\t', index=False)
    (out / 'dedup').mkdir(exist_ok=True)
    for fn in ('decisions.parquet', 'pairs_layer.tsv', 'pairs_set.tsv'):
        src = work / 'dedup' / fn
        if src.exists():
            (out / 'dedup' / fn).write_bytes(src.read_bytes())
    if (work / 'survey.tsv').exists():
        shutil.copy(work / 'survey.tsv', out / 'report_body_elements_survey.tsv')
    if (work / 'inventory.tsv').exists():
        shutil.copy(work / 'inventory.tsv', out / 'report_read_inventory.tsv')
    st = docs.groupby(['final', 'collection']).agg(documents=('doc_id', 'size'), words=('n_words', 'sum'), db_words=('db_words', 'sum')).reset_index()
    st.to_csv(out / 'report_by_final_status.tsv', sep='\t', index=False)
    docs.groupby('final').agg(documents=('doc_id', 'size'), db_words=('db_words', 'sum')).reset_index().to_csv(out / 'report_status_totals.tsv', sep='\t', index=False)
    docs.groupby(['collection', 'lang_db', 'final']).agg(documents=('doc_id', 'size'), db_words=('db_words', 'sum')).reset_index()\
        .to_csv(out / 'report_by_collection_lang_final.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
