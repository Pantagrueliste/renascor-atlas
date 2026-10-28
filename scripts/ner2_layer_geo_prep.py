"""NER layer v1, geocoding stage 1 (CPU, Hypatia): the distinct written forms of the recommended spans.

Keeps the spans with confidence >= 0.98 and agreement of >= 2 of the 4 models (the "recommended" spans of ``spans.parquet``), cleans each surface
(Unicode NFC, long s, surrounding punctuation, possessive 's, white space), and counts the distinct (language, form) pairs.

    python scripts/ner2_layer_geo_prep.py --layer ~/renascor-atlas-data/ner/layer_v1 --out ~/renascor-atlas-data/ner/layer_v1/geo

Writes in --out:
    forms.parquet       one row per (lang, key): key, form (most frequent written form), variants (JSON list of [surface, n]), n, n_docs, n_cap (mentions written with a
                        capital), q0..q9 (mentions per 25 years from 1450), status (ok / dropped reason)
    doc_forms.parquet   (doc_id, lang, key, n): the mentions of each form per document
    prep_summary.json   counts (spans kept, dropped by reason)

A form is dropped before geocoding when it is shorter than 3 letters, holds a digit or has more than 6 words. The share of mentions written with a capital
(``cap_share``; documents whose spans are mostly lower case, i.e. lower-cased editions, count as capitalised) is kept as a column: early-modern printing writes
real places in lower case (westmyster, yorke), so it is used only against GeoNames-only matches (see ``ner2_layer_geo_points.py``).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PUNCT = '.,;:!?()[]{}<>"\'«»“”‘’„‚-—–*_/\\|~^`#=+&%$§¶†‡•·'
POSS = re.compile(r"[’'`]s$", re.I)
SPACE = re.compile(r'\s+')


def clean_surface(s: str) -> str:
    s = unicodedata.normalize('NFC', s or '').replace('ſ', 's').replace('­', '')
    s = s.replace('n~', 'ñ').replace('N~', 'Ñ')            # tilde written after the letter (Espan~a)
    s = SPACE.sub(' ', s).strip()
    s = s.strip(PUNCT + ' ')
    s = POSS.sub('', s).strip(PUNCT + ' ')
    s = re.sub(r'\s*-\s+', '', s)                 # "Lon- don" (line-break hyphen) -> "London"
    return s


def reason(s: str) -> str:
    letters = sum(ch.isalpha() for ch in s)
    if letters < 3:
        return 'short'
    if any(ch.isdigit() for ch in s):
        return 'digit'
    if len(s.split()) > 6:
        return 'long'
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-conf', type=float, default=0.98)
    ap.add_argument('--min-agree', type=int, default=2)
    ap.add_argument('--cap-share', type=float, default=0.0, help='forms written with a capital in fewer than this share of their mentions are dropped (default 0: kept, the share is a column; the points step uses it)')
    a = ap.parse_args()
    L = Path(a.layer).expanduser()
    O = Path(a.out).expanduser()
    O.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(L / 'spans.parquet', columns=['doc_id', 'surface', 'conf', 'n_agree', 'lang', 'year'])
    s = t.to_pandas()
    n0 = len(s)
    s = s[(s['conf'] >= a.min_conf) & (s['n_agree'] >= a.min_agree)].reset_index(drop=True)
    summ = {'spans_total': int(n0), 'spans_recommended': int(len(s))}
    uniq = s['surface'].unique()
    cm = {u: clean_surface(u) for u in uniq}
    s['form'] = s['surface'].map(cm)
    rs = {u: reason(c) for u, c in cm.items()}
    s['why'] = s['surface'].map(rs)
    s['cap'] = s['form'].str[:1].str.isupper() | s['form'].str[:1].str.isdigit()
    # a document whose spans are mostly lower case is a lower-cased edition: its spans say nothing about common nouns
    dshare = s.groupby('doc_id')['cap'].agg(['mean', 'size'])
    lower_docs = set(dshare.index[(dshare['mean'] < 0.5) & (dshare['size'] >= 5)])
    s['cap_raw'] = s['cap']
    s.loc[s['doc_id'].isin(lower_docs), 'cap'] = True
    summ['lowercased_documents'] = len(lower_docs)
    s['key'] = s['form'].str.casefold()
    bad = s['why'] != ''
    summ['spans_dropped_clean'] = {k: int(v) for k, v in s.loc[bad, 'why'].value_counts().items()}
    g = s[~bad]
    q = np.clip(((g['year'].astype('float64') - 1450) // 25).fillna(-1).astype(int), -1, 9)
    g = g.assign(q=q)
    g['n'] = 1
    base = g.groupby(['lang', 'key']).agg(n=('n', 'sum'), n_cap=('cap', 'sum'), n_docs=('doc_id', 'nunique')).reset_index()
    qc = g[g['q'] >= 0].groupby(['lang', 'key', 'q']).size().unstack(fill_value=0).reindex(columns=range(10), fill_value=0)
    qc.columns = [f'q{i}' for i in range(10)]
    base = base.merge(qc.reset_index(), on=['lang', 'key'], how='left').fillna({f'q{i}': 0 for i in range(10)})
    vc = g.groupby(['lang', 'key', 'form']).size().reset_index(name='c').sort_values(['lang', 'key', 'c', 'form'], ascending=[True, True, False, True])
    # prefer a capitalised written form for display; variants = the written forms (cleaned) with counts
    vc['cap'] = vc['form'].str[:1].str.isupper()
    top = vc.sort_values(['lang', 'key', 'cap', 'c'], ascending=[True, True, False, False]).drop_duplicates(['lang', 'key'])[['lang', 'key', 'form']]
    var = vc.groupby(['lang', 'key']).apply(lambda d: json.dumps([[f, int(c)] for f, c in zip(d['form'], d['c'])][:12], ensure_ascii=False), include_groups=False).reset_index(name='variants')
    base = base.merge(top, on=['lang', 'key']).merge(var, on=['lang', 'key'])
    base['cap_share'] = base['n_cap'] / base['n']
    base['status'] = np.where(base['cap_share'] < a.cap_share, 'lowercase', 'ok')
    for i in range(10):
        base[f'q{i}'] = base[f'q{i}'].astype('int32')
    base['n'] = base['n'].astype('int64')
    base['n_docs'] = base['n_docs'].astype('int64')
    base.to_parquet(O / 'forms.parquet', index=False)
    dfm = g.groupby(['doc_id', 'lang', 'key']).size().reset_index(name='n')
    dfm.to_parquet(O / 'doc_forms.parquet', index=False)
    summ['forms'] = int(len(base))
    summ['forms_ok'] = int((base['status'] == 'ok').sum())
    summ['forms_lowercase'] = int((base['status'] == 'lowercase').sum())
    summ['mentions_kept_after_clean'] = int(len(g))
    summ['mentions_in_lowercase_forms'] = int(base.loc[base['status'] == 'lowercase', 'n'].sum())
    summ['mentions_in_ok_forms'] = int(base.loc[base['status'] == 'ok', 'n'].sum())
    summ['by_lang'] = {k: {'forms': int(len(d)), 'forms_ok': int((d['status'] == 'ok').sum()), 'mentions': int(d['n'].sum()),
                           'mentions_ok': int(d.loc[d['status'] == 'ok', 'n'].sum())} for k, d in base.groupby('lang')}
    (O / 'prep_summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), 'utf-8')
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    for lg, d in base.groupby('lang'):
        print(lg, d.sort_values('n', ascending=False).head(25)[['form', 'n', 'n_docs', 'cap_share']].to_string(index=False))


if __name__ == '__main__':
    main()
