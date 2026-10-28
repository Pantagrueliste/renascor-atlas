"""Document table for the Atlas place cards: one row per document with at least one place mention.

Stages (all re-runnable, nothing is written outside the output directory):

  keys    mentions.parquet -> keys.parquet     one row per (coll_slug, source_path, doc_id): n_place_mentions,
                                               years / in_period / licence / tier / renascor_id from the mentions
  raw     keys + raw files -> raw/<slug>.parquet   header metadata of each document (docs_extract.py), as JSON
  build   raw + keys + rules -> documents.parquet, documents_summary.tsv

  python -I -m ratlas.docs_table keys  --mentions .../v1/mentions.parquet --out DIR/keys.parquet
  python -I -m ratlas.docs_table raw   SLUG [SLUG...] --keys DIR/keys.parquet --out DIR/raw [--sample N]
  python -I -m ratlas.docs_table build --keys DIR/keys.parquet --raw DIR/raw --out DIR/cards

Rules per collection (document type, title rule, URL pattern) live in docs_rules.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import common as C
from . import docs_extract as X

CATALOGUE_BASE = 'https://pantagrueliste.github.io/renascor/#'
PERIOD = (1450, 1700)


# ======================================================================================================================
# keys
# ======================================================================================================================
def make_keys(mentions: Path, out: Path) -> None:
    import pandas as pd
    cols = ['coll_slug', 'source_path', 'doc_id', 'kind', 'renascor_id', 'doc_year', 'doc_year_from', 'doc_year_to',
            'date_source', 'in_period', 'lang', 'licence', 'tier', 'zone', 'role']
    t = pq.read_table(mentions, columns=cols, filters=[('kind', '=', 'place')])
    df = t.to_pandas()
    g = df.groupby(['coll_slug', 'source_path', 'doc_id'], sort=True, dropna=False)
    keys = g.agg(n_place_mentions=('kind', 'size'),
                 year_from=('doc_year_from', 'min'), year_to=('doc_year_to', 'max'),
                 date_source=('date_source', 'first'),
                 renascor_id=('renascor_id', 'first'), lang_m=('lang', lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
                 licence=('licence', 'first'), tier=('tier', 'first'),
                 n_in_period=('in_period', lambda s: int((s == True).sum())),   # noqa: E712
                 n_out_period=('in_period', lambda s: int((s == False).sum())),  # noqa: E712
                 n_hdr=('zone', lambda s: int((s == 'hdr').sum())),
                 roles=('role', lambda s: ','.join(sorted(set(s))))).reset_index()
    keys['in_period'] = None
    keys.loc[keys.n_in_period > 0, 'in_period'] = True
    keys.loc[(keys.n_in_period == 0) & (keys.n_out_period > 0), 'in_period'] = False
    keys['doc_key'] = [doc_key(a, b, c) for a, b, c in zip(keys.coll_slug, keys.source_path, keys.doc_id)]
    assert keys.doc_key.is_unique, 'doc_key collision'
    out.parent.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(out, index=False)
    print(f'keys: {len(keys)} documents, {int(keys.n_place_mentions.sum())} place mentions -> {out}', file=sys.stderr)


def doc_key(slug: str, source_path: str, doc_id: str) -> str:
    """Short stable hash of the document identity (collection, raw path, document id)."""
    return hashlib.sha1(f'{slug}|{source_path}|{doc_id}'.encode('utf-8')).hexdigest()[:12]


# ======================================================================================================================
# raw
# ======================================================================================================================
def run_raw(slug: str, keys_path: Path, out_dir: Path, sample: int = 0) -> dict:
    import pandas as pd
    keys = pd.read_parquet(keys_path)
    k = keys[keys.coll_slug == slug]
    by_file: dict[str, set] = {}
    for sp, did in zip(k.source_path, k.doc_id):
        by_file.setdefault(sp, set()).add(did)
    paths = sorted(by_file)
    if sample and len(paths) > sample:
        step = len(paths) / sample
        paths = [paths[int(i * step)] for i in range(sample)]
    recs = [C.FileRecord(path=p, coll=slug) for p in paths]
    rows, errors, t0 = [], [], time.time()
    for r in C.iter_with_data(recs):
        try:
            got = {}
            for did, info in X.extract_file(slug, r.path, r.data, by_file[r.path]):
                got[did] = info
            for did in by_file[r.path]:
                if did in got:
                    rows.append((slug, r.path, did, X.jsonable(got[did])))
                elif sample == 0:
                    errors.append(f'{r.path} {did}: document not found in file')
        except Exception as e:   # noqa: BLE001
            errors.append(f'{r.path}: {type(e).__name__}: {e}')
        r.data = None
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = pa.table({'coll_slug': [r[0] for r in rows], 'source_path': [r[1] for r in rows],
                    'doc_id': [r[2] for r in rows], 'info': [r[3] for r in rows]},
                   schema=pa.schema([('coll_slug', pa.string()), ('source_path', pa.string()),
                                     ('doc_id', pa.string()), ('info', pa.string())]))
    pq.write_table(tab, out_dir / f'{slug}.parquet', compression='zstd')
    st = dict(slug=slug, files=len(paths), docs_expected=len(k) if not sample else None, docs=len(rows),
              errors=len(errors), seconds=round(time.time() - t0, 1), first_errors=errors[:20])
    (out_dir / f'{slug}.stats.json').write_text(json.dumps(st, indent=1))
    print(json.dumps({k2: st[k2] for k2 in ('slug', 'files', 'docs', 'errors', 'seconds')}), file=sys.stderr)
    return st


# ======================================================================================================================
# build
# ======================================================================================================================
COLUMNS = [
    ('doc_key', pa.string()), ('coll_slug', pa.string()), ('renascor_id', pa.string()),
    ('source_path', pa.string()), ('doc_id', pa.string()),
    ('title', pa.string()), ('title_source', pa.string()),
    ('date', pa.string()), ('date_precision', pa.string()), ('year_from', pa.int32()), ('year_to', pa.int32()),
    ('in_period', pa.bool_()), ('year_source', pa.string()), ('date_conflict', pa.bool_()),
    ('author', pa.string()), ('sender', pa.string()), ('addressee', pa.string()),
    ('place_written', pa.string()), ('place_received', pa.string()),
    ('doc_type', pa.string()), ('language', pa.string()), ('language_source', pa.string()),
    ('url', pa.string()), ('url_kind', pa.string()),
    ('catalogue_url', pa.string()), ('edition_url', pa.string()),
    ('repository', pa.string()),
    ('licence', pa.string()), ('tier', pa.string()), ('n_place_mentions', pa.int32()),
]
SCHEMA = pa.schema(COLUMNS)


def _catalogue(repo_entries: Path):
    import glob
    out = {}
    for p in glob.glob(str(repo_entries / '*.json')):
        e = json.loads(Path(p).read_text())
        out[Path(p).stem] = e
    return out


def build(keys_path: Path, raw_dir: Path, out_dir: Path, entries_dir: Path, slugs: list[str] | None = None,
          entities: Path | None = None) -> None:
    import pandas as pd
    from . import docs_rules as R
    keys = pd.read_parquet(keys_path)
    if entities and Path(entities).exists():
        ent = pq.read_table(entities, columns=['local_key', 'name'], filters=[('coll_slug', '=', 'ckcc_epistolarium')]).to_pandas()
        R.set_ckcc_entities(dict(zip(ent.local_key, ent.name)))
    cat = _catalogue(entries_dir)
    rows = []
    notes: dict = {}
    for p in sorted(raw_dir.glob('*.parquet')):
        slug = p.stem
        if slugs and slug not in slugs:
            continue
        raw = pq.read_table(p).to_pandas()
        k = keys[keys.coll_slug == slug].set_index(['source_path', 'doc_id'])
        for sp, did, info_s in zip(raw.source_path, raw.doc_id, raw['info']):
            kr = k.loc[(sp, did)]
            info = json.loads(info_s)
            rows.append(R.make_row(slug, sp, did, info, kr, cat))
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {n: [r.get(n) for r in rows] for n, _ in COLUMNS}
    tab = pa.table(cols, schema=SCHEMA)
    pq.write_table(tab, out_dir / 'documents.parquet', compression='zstd')
    df = tab.to_pandas()
    summary(df, keys, out_dir / 'documents_summary.tsv')
    print(f'documents: {len(df)} rows -> {out_dir}/documents.parquet', file=sys.stderr)


def summary(df, keys, out: Path) -> None:
    import pandas as pd
    g = df.groupby('coll_slug')
    s = pd.DataFrame({
        'documents': g.size(),
        'expected': keys.groupby('coll_slug').size(),
        'place_mentions': g.n_place_mentions.sum(),
        'with_title': g.title_source.apply(lambda x: round((x != 'doc_id').mean(), 4)),
        'title_synthetic': g.title_source.apply(lambda x: round((x == 'synthetic').mean(), 4)),
        'date_conflicts': g.date_conflict.sum(),
        'with_exact_date': g.date_precision.apply(lambda x: round(x.isin(['day']).mean(), 4)),
        'with_month_or_day': g.date_precision.apply(lambda x: round(x.isin(['day', 'month']).mean(), 4)),
        'with_year': g.year_from.apply(lambda x: round(x.notna().mean(), 4)),
        'with_author_or_sender': g.apply(lambda x: round((x.author.notna() | x.sender.notna()).mean(), 4)),
        'with_language': g.language.apply(lambda x: round(x.notna().mean(), 4)),
        'with_url': g.url.apply(lambda x: round(x.notna().mean(), 4)),
        'url_kinds': g.url_kind.apply(lambda x: ','.join(f'{k}:{v}' for k, v in x.value_counts().items())),
        'doc_types': g.doc_type.apply(lambda x: ','.join(f'{k}:{v}' for k, v in x.value_counts().items())),
        'in_period_docs': g.in_period.apply(lambda x: int((x == True).sum())),   # noqa: E712
        'tier': g.tier.first(),
        'licence': g.licence.first(),
    }).reset_index()
    s.to_csv(out, sep='\t', index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='ratlas.docs_table')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('keys')
    a.add_argument('--mentions', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    b = sub.add_parser('raw')
    b.add_argument('slugs', nargs='+')
    b.add_argument('--keys', type=Path, required=True)
    b.add_argument('--out', type=Path, required=True)
    b.add_argument('--sample', type=int, default=0)
    c = sub.add_parser('build')
    c.add_argument('--keys', type=Path, required=True)
    c.add_argument('--raw', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--entries', type=Path, required=True, help='Renascor data/entries directory (read-only)')
    c.add_argument('--slugs', nargs='*')
    c.add_argument('--entities', type=Path, default=None, help='entities.parquet (CKCC names)')
    args = ap.parse_args(argv)
    if args.cmd == 'keys':
        make_keys(args.mentions, args.out)
    elif args.cmd == 'raw':
        for s in args.slugs:
            run_raw(s, args.keys, args.out, args.sample)
    else:
        build(args.keys, args.raw, args.out, args.entries, args.slugs, args.entities)


if __name__ == '__main__':
    main()
