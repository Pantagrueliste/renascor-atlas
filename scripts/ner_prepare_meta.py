#!/usr/bin/env python
"""Small tables for the NER data job (run on the Mac; light: three columns of the mentions table, no text is read).

    python scripts/ner_prepare_meta.py --out /Users/clem/Projects/Papers/RQ/derived/atlas/ner/meta

  edition_meta.tsv   slug, renascor_ids, licence, tier, context_stored, no_derivatives, languages (catalogue), period, words
  doc_meta.tsv.gz    source_path, slug, doc_year (median of the mentions), lang (most frequent mention language)

Then ship them to Hypatia (scripts/ner_push.sh does it).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd  # noqa: E402

from ratlas.collections import discover  # noqa: E402
from ratlas.ner.langs import norm_lang  # noqa: E402

ENTRIES = Path('/Users/clem/Projects/Papers/Renascor/data/entries')
V23 = Path('/Users/clem/Projects/Papers/RQ/derived/atlas/v2.3')
LIC = Path(__file__).resolve().parents[1] / 'data' / 'licences.tsv'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    lic = {}
    if not LIC.exists():
        raise SystemExit(f'internal licence table not found: {LIC} (not part of the public checkout)')
    with open(LIC, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            lic[r['slug']] = r
    rows = []
    for slug, col in sorted(discover().items()):
        m = col.module
        langs, period, words = [], '', 0
        for rid in m.RENASCOR_IDS:
            p = ENTRIES / f'{rid}.json'
            if p.exists():
                d = json.loads(p.read_text(encoding='utf-8'))
                langs += [x for x in d.get('languages', []) if x not in langs]
                period = period or d.get('period', '')
                words += int(d.get('words') or 0)
        l = lic.get(slug, {})
        rows.append({'slug': slug, 'renascor_ids': ';'.join(m.RENASCOR_IDS), 'licence': m.LICENCE, 'tier': m.TIER,
                     'context_stored': l.get('context_stored', ''), 'no_derivatives': l.get('no_derivatives', ''),
                     'languages': ';'.join(langs), 'period': period, 'words': words})
    with open(a.out / 'edition_meta.tsv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    t = pd.read_parquet(V23 / 'mentions.parquet', columns=['coll_slug', 'source_path', 'doc_year', 'lang', 'zone', 'kind'])
    t = t[t.zone == 'text']
    t['lang'] = t['lang'].map(lambda x: norm_lang(x) if isinstance(x, str) else None)
    yr = t.groupby('source_path').agg(slug=('coll_slug', 'first'), doc_year=('doc_year', 'median'))
    lg = (t.dropna(subset=['lang']).groupby(['source_path', 'lang']).size().reset_index(name='n')
          .sort_values(['source_path', 'n', 'lang'], ascending=[True, False, True]).drop_duplicates('source_path')
          .set_index('source_path')['lang'])
    out = yr.join(lg.rename('lang')).reset_index()
    with gzip.open(a.out / 'doc_meta.tsv.gz', 'wt', encoding='utf-8', newline='') as fh:
        out.to_csv(fh, sep='\t', index=False)
    print(f'{len(rows)} editions, {len(out)} documents with text mentions, '
          f'{out.lang.notna().sum()} with a mention language, {out.doc_year.notna().sum()} with a year')
    return 0


if __name__ == '__main__':
    sys.exit(main())
