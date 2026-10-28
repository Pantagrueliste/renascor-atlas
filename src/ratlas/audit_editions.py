"""The per-edition checks of the audit, re-run on a FRESH sample of v2 (3 source files and the 10 top places of each edition).

    python -m ratlas audit editions --all out_v2/all --final out_v2/reconcile/final --raw raw --out audit_v2/editions [--seed fresh-20261017]

Documents. For every edition, 3 source files drawn at random (seeded; the audit's own samples are not reused): the number of tagged
elements in the raw file (``persName placeName orgName geogName rs name``) against the mention rows of the file (all, default, excluded), and the
dates of the header (``date`` elements, years 1400-1750) against the date range the row carries.

Places. For every edition, the 10 places with the most default mentions: name, accepted QID and its label, the point, its source and kind, the
distance to the nearest same-name GeoNames feature (``gn_final_dist.tsv`` of check 4), the audit's name relation between name and label.
Flags: ``unplaced``, ``name_vs_label`` (the audit matcher finds no relation), ``far_namesake`` (> 50 km), ``area_point``, ``suggested_only``.
The flagged rows are read by hand; the TSVs are the working material, the verdict is written in the report.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import audit_compat as AC
from . import common as C

TAGS = ('persName', 'placeName', 'orgName', 'geogName', 'rs', 'name')
YEAR = re.compile(r'(?<!\d)(1[4-7]\d\d)(?!\d)')
DATE_ATTRS = ('when', 'notBefore', 'notAfter', 'from', 'to', 'when-iso', 'notBefore-iso', 'notAfter-iso')


def _t(msg):
    print(f'[audit_editions {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def raw_profile(data: bytes, path: str) -> dict:
    """Tag counts and header years of a raw file; ``fmt`` says whether the file could be read as XML."""
    low = path.lower().split('!')[-1]
    if not low.endswith(('.xml', '.tei', '.xhtml', '.html', '.htm')):
        return {'fmt': 'not-xml'}
    root = C.parse_xml(data)
    if root is None:
        return {'fmt': 'unparsable'}
    cnt = {t: 0 for t in TAGS}
    years = set()
    in_header = False
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        n = C.ln(el.tag)
        if n in cnt:
            cnt[n] += 1
        if n in ('date', 'origDate', 'creation', 'docDate'):
            for a in DATE_ATTRS:
                v = el.get(a)
                if v:
                    years.update(int(y) for y in YEAR.findall(v))
            years.update(int(y) for y in YEAR.findall(''.join(el.itertext())[:60]))
    core = sum(cnt[t] for t in ('persName', 'placeName', 'orgName', 'geogName'))
    return {'fmt': 'xml', 'tags': cnt, 'n_tags': int(sum(cnt.values())), 'n_core': int(core), 'years': sorted(years)}


def sample_files(m: pd.DataFrame, slug: str, seed: str, k: int = 3) -> list:
    files = sorted(m.loc[m['coll_slug'] == slug, 'source_path'].unique())
    rng = random.Random(f'{seed}-{slug}')
    return rng.sample(files, min(k, len(files)))


def file_row(m: pd.DataFrame, slug: str, path: str, raw_root: Path) -> dict:
    x = m[(m['coll_slug'] == slug) & (m['source_path'] == path)]
    default = x[~x['excluded'].fillna(False).astype(bool)]
    r = dict(coll_slug=slug, source_path=path, rows=len(x), rows_default=len(default), rows_excluded=len(x) - len(default),
             persons=int((default['kind'] == 'person').sum()), places=int((default['kind'] == 'place').sum()),
             zones=json.dumps(x['zone'].value_counts().to_dict()), docs=int(x['doc_id'].nunique()),
             year_from=x['doc_year_from'].min(), year_to=x['doc_year_to'].max(), date_sources='|'.join(sorted(x['date_source'].dropna().unique())))
    try:
        data = C.read_raw(path)
        p = raw_profile(data, path)
    except Exception as e:      # noqa: BLE001
        p = {'fmt': f'unreadable: {type(e).__name__}'}
    r['raw_fmt'] = p.get('fmt')
    r['raw_tags'] = p.get('n_tags')
    r['raw_core_tags'] = p.get('n_core')
    r['raw_tag_counts'] = json.dumps(p.get('tags')) if p.get('tags') else ''
    ys = p.get('years') or []
    r['header_years'] = ' '.join(map(str, ys[:12]))
    flags = []
    if p.get('fmt') == 'xml':
        nt, core = p['n_tags'], p['n_core']
        if nt >= 3 and len(x) == 0:
            flags.append('tags_but_no_rows')
        elif core >= 10 and r['docs'] == 1 and len(x) < 0.5 * core:
            flags.append('rows_below_core_tags')
        elif core >= 10 and r['docs'] == 1 and len(x) > 1.5 * core + 5:
            flags.append('rows_above_core_tags')
        elif len(x) >= 20 and nt == 0:
            flags.append('rows_without_tags')
        if ys and pd.notna(r['year_from']) and pd.notna(r['year_to']):
            lo, hi = r['year_from'], r['year_to']
            if not any(lo - 1 <= y <= hi + 1 for y in ys) and r['docs'] == 1 and hi - lo < 30:
                flags.append('date_not_in_header_years')
    r['flags'] = ','.join(flags)
    return r


def top_places(pl: pd.DataFrame, dist: pd.DataFrame, slug: str, n: int = 10, wdnames: Optional[dict] = None) -> pd.DataFrame:
    x = pl[(pl['coll_slug'] == slug) & (pl['scope'] != 'excluded') & (pl['n_mentions'] > 0)].sort_values('n_mentions', ascending=False).head(n)
    x = x.merge(dist[['coll_slug', 'entity_key', 'd_km', 'nearest_cc']], on=['coll_slug', 'entity_key'], how='left')
    rows = []
    for r in x.itertuples(index=False):
        fl = []
        if pd.isna(r.lat):
            fl.append('unplaced')
        if isinstance(r.qid, str) and isinstance(r.wd_label, str):
            ek = AC.name_keys(r.name, [])
            labels = (wdnames or {}).get(r.qid) or [r.wd_label]
            rel = AC.relation(ek, AC.auth_keys(labels))[0]
            if rel.startswith('none'):
                fl.append('name_vs_label')
        if pd.notna(r.d_km) and r.d_km > 50 and r.coord_kind == 'point' and r.coord_source != 'name_exonym':
            fl.append('far_namesake')
        if r.coord_kind == 'area':
            fl.append('area_point')
        if not isinstance(r.qid, str) and isinstance(r.qid_suggested, str):
            fl.append('suggested_only')
        rows.append(dict(coll_slug=slug, name=r.name, n=int(r.n_mentions), qid=r.qid, wd_label=r.wd_label, lat=r.lat, lon=r.lon,
                         coord_source=r.coord_source, kind=r.coord_kind, place_kind=r.place_kind, d_km=r.d_km, gc_conf=r.gc_conf,
                         gc_evidence=(str(r.gc_evidence)[:90] if isinstance(r.gc_evidence, str) else ''), note_type=r.note_type,
                         flags=','.join(fl)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas audit editions')
    ap.add_argument('--all', type=Path, required=True)
    ap.add_argument('--final', type=Path, required=True)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', default='fresh-20261017')
    ap.add_argument('--dist', type=Path, default=None, help='gn_final_dist.tsv of check 4')
    ap.add_argument('--only', default=None, help='comma-separated slugs (debugging)')
    ap.add_argument('--rdir', type=Path, default=None, help='reconcile_v2 (wd_entities: all the labels of an item, not only the English one)')
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ['RATLAS_RAW'] = str(a.raw)
    m = pq.read_table(a.all / 'mentions.parquet', columns=['coll_slug', 'source_path', 'doc_id', 'doc_year_from', 'doc_year_to', 'date_source',
                                                            'kind', 'zone', 'excluded']).to_pandas()
    pl = pq.read_table(a.final / 'places_local_geo.parquet', columns=['coll_slug', 'entity_key', 'name', 'n_mentions', 'scope', 'qid', 'qid_suggested',
                                                                      'wd_label', 'lat', 'lon', 'coord_source', 'coord_kind', 'place_kind', 'gc_conf',
                                                                      'gc_evidence', 'note_type']).to_pandas()
    dist = pd.read_csv(a.dist, sep='\t') if a.dist and a.dist.exists() else pd.DataFrame(columns=['coll_slug', 'entity_key', 'd_km', 'nearest_cc'])
    wdnames = None
    if a.rdir and (a.rdir / 'wd_entities.parquet').exists():
        w = pq.read_table(a.rdir / 'wd_entities.parquet', columns=['qid', 'labels', 'aliases']).to_pandas()
        w = w[w['qid'].isin(set(pl['qid'].dropna()))]
        wdnames = {}
        for q, lab, ali in zip(w['qid'], w['labels'], w['aliases']):
            lab = json.loads(lab) if lab else {}
            ali = json.loads(ali) if ali else {}
            wdnames[q] = list(dict.fromkeys(list(lab.values()) + [x for v in ali.values() for x in v]))
    slugs = sorted(m['coll_slug'].unique())
    if a.only:
        slugs = [x for x in slugs if x in a.only.split(',')]
    docs, places = [], []
    for s in slugs:
        for p in sample_files(m, s, a.seed):
            docs.append(file_row(m, s, p, a.raw))
        places.append(top_places(pl, dist, s, wdnames=wdnames))
        _t(f'{s}')
    D = pd.DataFrame(docs)
    P = pd.concat(places, ignore_index=True)
    D.to_csv(a.out / 'edition_sample_files.tsv', sep='\t', index=False)
    P.to_csv(a.out / 'edition_sample_places.tsv', sep='\t', index=False)
    summ = {'editions': len(slugs), 'files': len(D), 'files_flagged': int((D['flags'] != '').sum()), 'places': len(P),
            'places_flagged': int((P['flags'].fillna('') != '').sum()), 'seed': a.seed}
    (a.out / 'edition_sample_summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    print(json.dumps(summ))
    return 0


if __name__ == '__main__':
    sys.exit(main())
