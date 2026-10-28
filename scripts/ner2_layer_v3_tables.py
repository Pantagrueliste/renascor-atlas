"""NER layer v3, summary tables (CPU, Hypatia or Mac): spans, recommended, dropped as overlapping editor tags, placed mentions and points, by collection.

    python scripts/ner2_layer_v3_tables.py --v3 ~/renascor-atlas-data/ner/layer_v3 --out DIR

Needs the outputs of ner2_layer_v3_merge.py (``tei/spans.parquet``, ``extra/spans.parquet``, ``<set>/docs.parquet``) and, for the placed columns, of the geocoding steps
(``geo/points/forms_placed.parquet``, ``geo/doc_forms.parquet``, ``geo/ctx_map.parquet``). Writes in --out:
  v3_by_collection.tsv        one row per (set, collection, language): documents, documents run, words, spans, recommended, recommended dropped as overlapping an editor tag
                              (and by tag kind), recommended kept, cleaned mentions, placed mentions, points, documents with a placed mention
  v3_dropped_by_tag.tsv       spans overlapping an editor tag, by tag element, type and kind (all spans / recommended)
  v3_placed_by_method.tsv     placed mentions by method of the coordinates, and those placed by propagation from a place of the same edition
  v3_totals.json              totals
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def by_collection(sp: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    d = docs.groupby(['set', 'collection', 'lang']).agg(docs=('doc_id', 'size'), docs_run=('run', 'sum'), words=('n_words', 'sum')).reset_index()
    sp = sp.copy()
    for k in ('place', 'person', 'org', 'other'):
        sp['rd_' + k] = sp['recommended'] & sp['ed_overlap'] & (sp['ed_kind'] == k)
    sp['rd'] = sp['recommended'] & sp['ed_overlap']
    g = sp.groupby(['set', 'collection', 'lang']).agg(spans=('conf', 'size'), recommended=('recommended', 'sum'), dropped_ed=('rd', 'sum'), dropped_ed_place=('rd_place', 'sum'),
                                                      dropped_ed_person=('rd_person', 'sum'), dropped_ed_org=('rd_org', 'sum'), dropped_ed_other=('rd_other', 'sum'),
                                                      rec_final=('rec_final', 'sum'), spans_overlap_all=('ed_overlap', 'sum')).reset_index()
    return d.merge(g, on=['set', 'collection', 'lang'], how='left').fillna(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sets', nargs='*', default=['tei', 'extra', 'seen'])
    ap.add_argument('--geo-name', default='geo')
    a = ap.parse_args()
    v3 = a.v3.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    sp = pd.concat([pd.read_parquet(v3 / s / 'spans.parquet', columns=['doc_id', 'conf', 'recommended', 'ed_overlap', 'ed_tag', 'ed_type', 'ed_kind', 'rec_final', 'lang', 'collection', 'set'])
                    for s in a.sets], ignore_index=True)
    docs = pd.concat([pd.read_parquet(v3 / s / 'docs.parquet') for s in a.sets], ignore_index=True)
    tab = by_collection(sp, docs)
    tot = {'spans': int(len(sp)), 'recommended': int(sp['recommended'].sum()), 'dropped_ed': int((sp['recommended'] & sp['ed_overlap']).sum()), 'rec_final': int(sp['rec_final'].sum())}
    # dropped by tag
    ov = sp[sp['ed_overlap']].copy()
    ov['ed_type'] = ov['ed_type'].fillna('')
    dt = ov.groupby(['ed_kind', 'ed_tag', 'ed_type']).agg(spans=('conf', 'size'), recommended=('recommended', 'sum')).reset_index().sort_values('spans', ascending=False)
    dt.to_csv(a.out / 'v3_dropped_by_tag.tsv', sep='\t', index=False)
    dk = ov.groupby('ed_kind').agg(spans=('conf', 'size'), recommended=('recommended', 'sum')).reset_index()
    dk.to_csv(a.out / 'v3_dropped_by_kind.tsv', sep='\t', index=False)
    fp = v3 / a.geo_name / 'points' / 'forms_placed.parquet'
    if fp.exists():
        F = pd.read_parquet(fp)
        df = pd.read_parquet(v3 / a.geo_name / 'doc_forms.parquet')
        df = df.merge(docs[['doc_id', 'collection', 'set']], on='doc_id', how='left')
        m = df.merge(F[['lang', 'key', 'fate', 'pid', 'gc_source', 'gc_cand', 'gc_conf']], on=['lang', 'key'], how='left')
        m['placed'] = m['fate'] == 'placed'
        m['pl_n'] = np.where(m['placed'], m['n'], 0)
        g = m.groupby(['set', 'collection', 'lang']).agg(mentions_clean=('n', 'sum'), mentions_placed=('pl_n', 'sum')).reset_index()
        pts = m[m['placed']].groupby(['set', 'collection', 'lang'])['pid'].nunique().rename('points').reset_index()
        dw = m[m['placed']].groupby(['set', 'collection', 'lang'])['doc_id'].nunique().rename('docs_with_placed').reset_index()
        tab = tab.merge(g, on=['set', 'collection', 'lang'], how='left').merge(pts, on=['set', 'collection', 'lang'], how='left').merge(dw, on=['set', 'collection', 'lang'], how='left').fillna(0)
        # points per collection counted over the whole layer: a point may serve several collections
        tot['mentions_clean'] = int(m['n'].sum())
        tot['mentions_placed'] = int(m['pl_n'].sum())
        tot['points'] = int(m.loc[m['placed'], 'pid'].nunique())
        for s in a.sets:
            mm = m[(m['set'] == s) & m['placed']]
            tot[f'points_{s}'] = int(mm['pid'].nunique())
            tot[f'mentions_placed_{s}'] = int(mm['n'].sum())
        # method; same-edition propagation
        try:
            cm = pd.read_parquet(v3 / a.geo_name / 'ctx_map.parquet')
            m = m.merge(cm, on=['lang', 'key'], how='left')
            m['cand_coll'] = m['gc_cand'].fillna('').str.split(':').str[0]
            m['same_edition'] = (m['gc_source'] == 'name_propagation') & (m['ctx'].fillna('') != '') & (m['cand_coll'] == m['ctx'])
            m['ctx'] = m['ctx'].fillna('')
            pm = m[m['placed']].groupby(['set', 'gc_source']).agg(mentions=('n', 'sum'), forms=('key', 'nunique')).reset_index()
            se = m[m['placed'] & m['same_edition']].groupby('set')['n'].sum()
            pm['of_which_same_edition_place'] = [int(se.get(s, 0)) if g == 'name_propagation' else 0 for s, g in zip(pm['set'], pm['gc_source'])]
            pm.to_csv(a.out / 'v3_placed_by_method.tsv', sep='\t', index=False)
            ce = m[m['placed']].groupby(['set', 'ctx']).agg(mentions=('n', 'sum')).reset_index()
            ce.to_csv(a.out / 'v3_placed_by_context_edition.tsv', sep='\t', index=False)
            tot['placed_mentions_propagation_same_edition'] = int(m.loc[m['placed'] & m['same_edition'], 'n'].sum())
        except FileNotFoundError:
            pass
    for c in tab.columns:
        if c not in ('set', 'collection', 'lang'):
            tab[c] = tab[c].astype('int64')
    tab = tab.sort_values(['set', 'words'], ascending=[True, False])
    tab.to_csv(a.out / 'v3_by_collection.tsv', sep='\t', index=False)
    (a.out / 'v3_totals.json').write_text(json.dumps(tot, indent=1), 'utf-8')
    print(tab.to_string())
    print(json.dumps(tot, indent=1))


if __name__ == '__main__':
    main()
