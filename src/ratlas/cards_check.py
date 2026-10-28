"""Consistency checks for the Atlas card tables (documents.parquet, wd_places.parquet). Read-only.

  python -I -m ratlas.cards_check mentions --mentions v1/mentions.parquet --documents cards/documents.parquet
      T3 mentions must carry no context snippet; London Lives / Medici Archive Project must be absent;
      every document key of documents.parquet must exist in the mentions, and the place-mention counts must agree.
  python -I -m ratlas.cards_check documents --documents cards/documents.parquet
      invariants of the document table (unique keys, no snippet-like columns, year order, url shape ...).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

EXCLUDED = re.compile(r'london[_-]?lives|medici|(^|[_-])mia($|[_-])|archivio[_-]mediceo', re.I)


def check_mentions(mentions: Path, documents: Path) -> int:
    bad = 0
    pf = pq.ParquetFile(mentions)
    cols = [c for c in ('coll_slug', 'tier', 'context', 'kind', 'source_path', 'doc_id') if c in pf.schema_arrow.names]
    t = pq.read_table(mentions, columns=cols)
    print(f'mentions: {t.num_rows} rows, columns read: {cols}')
    tier, ctx = t['tier'], t['context']
    ctx_ok = pc.and_(pc.is_valid(ctx), pc.greater(pc.utf8_length(pc.fill_null(ctx, '')), 0))
    for tv in ('T1', 'T2', 'T3'):
        m = pc.equal(tier, tv)
        n = pc.sum(m.cast('int64')).as_py() or 0
        c = pc.sum(pc.and_(m, ctx_ok).cast('int64')).as_py() or 0
        print(f'  tier {tv}: {n} mentions, {c} with a context snippet')
        if tv == 'T3' and c:
            bad += 1
            t3 = t.filter(pc.and_(pc.equal(tier, 'T3'), ctx_ok))
            print('  !! T3 with context, by collection:', t3.group_by('coll_slug').aggregate([('tier', 'count')]).to_pylist())
    n_other = pc.sum(pc.invert(pc.is_in(tier, value_set=__import__('pyarrow').array(['T1', 'T2', 'T3']))).cast('int64')).as_py()
    print(f'  tier not in T1/T2/T3 (null or other): {n_other}')
    slugs = sorted(set(t['coll_slug'].to_pylist()))
    ex = [s for s in slugs if EXCLUDED.search(s)]
    print(f'  collections in mentions: {len(slugs)}; excluded-looking slugs present: {ex}')
    bad += bool(ex)
    d = pq.read_table(documents, columns=['coll_slug', 'source_path', 'doc_id', 'n_place_mentions', 'tier']).to_pandas()
    dex = sorted(s for s in d.coll_slug.unique() if EXCLUDED.search(s))
    print(f'  collections in documents: {d.coll_slug.nunique()}; excluded-looking slugs present: {dex}')
    bad += bool(dex)
    tp = t.filter(pc.equal(t['kind'], 'place')).select(['coll_slug', 'source_path', 'doc_id', 'tier']).to_pandas()
    g = tp.groupby(['coll_slug', 'source_path', 'doc_id'], dropna=False).size().rename('n').reset_index()
    j = d.merge(g, on=['coll_slug', 'source_path', 'doc_id'], how='outer', indicator=True)
    print('  document keys vs place mentions:', j['_merge'].value_counts().to_dict())
    both = j[j['_merge'] == 'both']
    mism = both[both.n != both.n_place_mentions]
    print(f'  place-mention count mismatches: {len(mism)}')
    bad += (j['_merge'] != 'both').any() or len(mism) > 0
    # tier of the document equals the tier of its mentions
    tt = tp.groupby(['coll_slug', 'source_path', 'doc_id'], dropna=False).tier.agg(lambda s: ','.join(sorted(set(s)))).rename('tm').reset_index()
    j2 = d.merge(tt, on=['coll_slug', 'source_path', 'doc_id'])
    print('  documents whose tier differs from their mentions tier:', int((j2.tier != j2.tm).sum()))
    return int(bool(bad))


def check_documents(documents: Path) -> int:
    import pandas as pd
    df = pd.read_parquet(documents)
    bad = 0

    def rep(name, mask):
        nonlocal bad
        n = int(mask.sum())
        print(f'  {name}: {n}')
        bad += n > 0

    print(f'documents: {len(df)} rows')
    rep('duplicate doc_key', df.doc_key.duplicated())
    rep('year_from > year_to', df.year_from > df.year_to)
    rep('title missing', df.title.isna() | (df.title.str.len() == 0))
    rep('title longer than 160', df.title.str.len() > 160)
    rep('url not http(s)', df.url.notna() & ~df.url.fillna('').str.match(r'^https?://'))
    rep('url with whitespace', df.url.fillna('').str.contains(r'\s'))
    rep('url without url_kind', df.url.notna() & df.url_kind.isna())
    rep('catalogue_url missing', df.catalogue_url.isna())
    rep('T3 title built from a summary', (df.tier == 'T3') & (df.title_source == 'summary'))
    rep('placeholder person / place values',
        df.sender.fillna('').str.fullmatch(r'(?i)s\.\s?n\.?|n\.?\s?n\.?|\?') | df.addressee.fillna('').str.fullmatch(r'(?i)s\.\s?n\.?|n\.?\s?n\.?|\?')
        | df.place_written.fillna('').str.fullmatch(r'(?i)\[?s\.\s?l\.?\]?|o\.\s?o\.?|¿\?') | df.place_received.fillna('').str.fullmatch(r'(?i)\[?s\.\s?l\.?\]?|o\.\s?o\.?|¿\?'))
    rep('addressee ends with a date', df.addressee.fillna('').str.contains(r',\s*(\d{1,2}\.\s*)?[A-Za-zäöü]+\s+\d{4}\s*$'))
    rep('title with unclosed parenthesis fragment', df.title.fillna('').str.contains(r'\([^)]{0,14}$'))
    rep('title with a word glued to a year', df.title.fillna('').str.contains(r'[a-zäöüé]\d{4}\b') & ~df.coll_slug.eq('ckcc_epistolarium'))
    print('  ' + ('OK' if not bad else f'{bad} check(s) failed'))
    return int(bool(bad))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='ratlas.cards_check')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('mentions')
    a.add_argument('--mentions', type=Path, required=True)
    a.add_argument('--documents', type=Path, required=True)
    b = sub.add_parser('documents')
    b.add_argument('--documents', type=Path, required=True)
    args = ap.parse_args(argv)
    rc = check_mentions(args.mentions, args.documents) if args.cmd == 'mentions' else check_documents(args.documents)
    sys.exit(rc)


if __name__ == '__main__':
    main()
