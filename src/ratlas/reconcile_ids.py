"""Reconcile stage, part 1: authority-id normalisation and collection.

    python -m ratlas reconcile ids [--v1 DIR] [--dest DIR]

Reads canonical person and place mentions and register entities of ``mentions.parquet`` / ``entities.parquet`` and
writes ``ids.tsv`` (kind, scheme, id, n_mentions, n_in_period, n_register): every distinct authority id after
normalisation. ``norm_id`` is shared with the build stage so that an id maps to the same key everywhere.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# scheme -> Wikidata property. 'wikipedia' (sitelinks) and 'wikidata' itself are handled separately.
PROPS = {
    'gnd': 'P227', 'viaf': 'P214', 'geonames': 'P1566', 'tgn': 'P1667', 'ulan': 'P245', 'pleiades': 'P1584',
    'hls': 'P902', 'isni': 'P213', 'loc': 'P244', 'cerl': 'P1871', 'odnb': 'P1415', 'deutsche-biographie': 'P7902',
    'bnf': 'P268',
}
# Order in which ids vote for / identify an entity (strongest first).
PRIORITY = ('wikidata', 'geonames', 'tgn', 'pleiades', 'gnd', 'hls', 'viaf', 'ulan', 'isni', 'loc', 'cerl', 'odnb',
            'deutsche-biographie', 'bnf', 'wikipedia')

_RE = {
    'wikidata': re.compile(r'^Q[1-9]\d{0,10}$'),
    'gnd': re.compile(r'^\d{1,10}-?[\dX]$'),
    'viaf': re.compile(r'^[1-9]\d{0,11}$'),
    'geonames': re.compile(r'^[1-9]\d{0,9}$'),
    'tgn': re.compile(r'^\d{7}$'),
    'ulan': re.compile(r'^\d{9}$'),
    'pleiades': re.compile(r'^\d{3,9}$'),
    'hls': re.compile(r'^\d{1,6}$'),
    'isni': re.compile(r'^\d{15}[\dX]$'),
    'loc': re.compile(r'^[a-z]{1,2}\d{6,12}$'),
    'cerl': re.compile(r'^cn[ipc]\d{8}$'),
    'odnb': re.compile(r'^\d{1,6}$'),
    'deutsche-biographie': re.compile(r'^\d{8,9}[\dX]?$'),
    'bnf': re.compile(r'^\d{8}[0-9a-z]$'),
}


def norm_id(scheme: str, raw: Optional[str]) -> Optional[tuple[str, str]]:
    """(scheme, raw) -> (canonical scheme, canonical id), or None if unusable. Handles the 'other' schemes
    (``url``, ``wikipedia``...) of the tables."""
    if raw is None:
        return None
    v = str(raw).strip()
    if not v:
        return None
    s = scheme
    if s == 'url':
        u = v
        m = re.search(r'wikidata\.org/(?:wiki|entity)/(Q\d+)', u)
        if m:
            return ('wikidata', m.group(1))
        m = re.search(r'viaf\.org/(?:viaf/)?(\d+)', u)
        if m:
            s, v = 'viaf', m.group(1)
        else:
            m = re.search(r'd-nb\.info/gnd/([0-9X-]+)', u)
            if m:
                s, v = 'gnd', m.group(1)
            else:
                m = re.search(r'sws\.geonames\.org/(\d+)|geonames\.org/(\d+)', u)
                if m:
                    s, v = 'geonames', m.group(1) or m.group(2)
                else:
                    return None
    if s == 'wikipedia':
        if ':' not in v:
            return None
        lang, _, title = v.partition(':')
        title = unquote(title).replace('_', ' ').strip()
        lang = lang.strip().lower()
        if not re.fullmatch(r'[a-z]{2,3}(-[a-z]+)?', lang) or not title:
            return None
        return ('wikipedia', f'{lang}:{title}')
    if s == 'wikidata':
        v = v.upper()
        v = re.sub(r'^.*?(Q\d+).*$', r'\1', v) if 'Q' in v else v
        return ('wikidata', v) if _RE['wikidata'].match(v) else None
    if s == 'gnd':
        v = re.sub(r'^(https?://d-nb\.info/gnd/|gnd:)', '', v, flags=re.I).upper()
    elif s == 'viaf':
        v = re.sub(r'^(https?://(www\.)?viaf\.org/(viaf/)?|viaf:)', '', v, flags=re.I).rstrip('/')
    elif s == 'hls':
        v = v.lstrip('0') or '0'
    elif s == 'isni':
        v = re.sub(r'[\s-]', '', v).upper()
    elif s == 'bnf':
        v = re.sub(r'^cb', '', v.lower())
    elif s == 'deutsche-biographie':
        v = re.sub(r'^gnd', '', v.lower()).upper()
    elif s == 'cerl':
        v = v.lower()
    elif s == 'loc':
        v = v.lower().replace(' ', '')
    elif s == 'tgn':
        v = re.sub(r'^tgn[: ]?', '', v, flags=re.I)
    if s in _RE and _RE[s].match(v):
        return (s, v)
    return None


def query_forms(scheme: str, idv: str) -> list[str]:
    """Strings to look up in the Wikidata statement of the property for one normalised id."""
    if scheme == 'hls':
        return list(dict.fromkeys([idv, idv.zfill(6)]))
    if scheme == 'isni':
        return [' '.join(idv[i:i + 4] for i in range(0, 16, 4)), idv]
    if scheme == 'bnf':
        return [idv]
    return [idv]


def back_form(scheme: str, found: str) -> str:
    """Wikidata value -> normalised id (inverse of query_forms)."""
    n = norm_id(scheme, found)
    return n[1] if n else found


# ----------------------------------------------------------------------------------------------------------------------
def _col(arr, name):
    return arr.field(name).to_pandas()


def _long_ids(tbl: pa.Table, extra: dict) -> pd.DataFrame:
    """One row per (row, scheme, raw id) from the auth struct of ``tbl``; ``extra`` = columns carried along."""
    auth = tbl.column('auth').combine_chunks()
    base = pd.DataFrame({k: tbl.column(k).to_pandas() for k in extra})
    parts = []
    for k in ('gnd', 'viaf', 'wikidata', 'geonames', 'tgn', 'ulan', 'pleiades', 'hls'):
        s = _col(auth, k)
        m = s.notna()
        if m.any():
            d = base[m.values].copy()
            d['scheme'] = k
            d['raw'] = s[m].values
            parts.append(d)
    other = auth.field('other')
    ln = pc.list_value_length(other).to_numpy(zero_copy_only=False)
    idx = (ln > 0).nonzero()[0]
    if len(idx):
        sub = other.take(pa.array(idx))
        flat = pc.list_flatten(sub).to_pylist()
        lens = pc.list_value_length(sub).to_numpy(zero_copy_only=False)
        rep = base.iloc[idx.repeat(lens)]
        d = rep.copy()
        sch, raw = zip(*[(x.partition(':')[0], x.partition(':')[2]) for x in flat])
        d['scheme'] = sch
        d['raw'] = raw
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=list(extra) + ['scheme', 'raw'])
    return pd.concat(parts, ignore_index=True)


def normalise_long(d: pd.DataFrame) -> pd.DataFrame:
    """Add normalised ``scheme`` / ``id`` columns; drop unusable rows."""
    pairs = {}
    for s, r in set(zip(d['scheme'], d['raw'])):
        pairs[(s, r)] = norm_id(s, r)
    n = [pairs[(s, r)] for s, r in zip(d['scheme'], d['raw'])]
    d = d.assign(nscheme=[x[0] if x else None for x in n], nid=[x[1] if x else None for x in n])
    return d[d['nid'].notna()].drop(columns=['scheme', 'raw']).rename(columns={'nscheme': 'scheme', 'nid': 'id'})


def collect(v1: Path, dest: Path, log=print) -> pd.DataFrame:
    t0 = time.time()
    m = pq.read_table(v1 / 'mentions.parquet', columns=['kind', 'canonical', 'in_period', 'auth'])
    m = m.filter(pc.and_(pc.equal(m.column('canonical'), True), pc.is_in(m.column('kind'), pa.array(['person', 'place']))))
    log(f'mentions: {m.num_rows} canonical person/place rows ({time.time() - t0:.0f}s)')
    m = m.append_column('ip', pc.fill_null(pc.equal(m.column('in_period'), True), False))
    d = normalise_long(_long_ids(m, {'kind': None, 'ip': None}))
    g = d.groupby(['kind', 'scheme', 'id']).agg(n_mentions=('ip', 'size'), n_in_period=('ip', 'sum')).reset_index()
    e = pq.read_table(v1 / 'entities.parquet', columns=['kind', 'auth'])
    e = e.filter(pc.is_in(e.column('kind'), pa.array(['person', 'place'])))
    de = normalise_long(_long_ids(e, {'kind': None}))
    ge = de.groupby(['kind', 'scheme', 'id']).size().rename('n_register').reset_index()
    out = g.merge(ge, how='outer', on=['kind', 'scheme', 'id'])
    for c in ('n_mentions', 'n_in_period', 'n_register'):
        out[c] = out[c].fillna(0).astype('int64')
    out = out.sort_values(['kind', 'scheme', 'n_mentions'], ascending=[True, True, False])
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / 'ids.tsv', sep='\t', index=False)
    log(f'{len(out)} distinct (kind, scheme, id) -> {dest / "ids.tsv"} ({time.time() - t0:.0f}s)')
    piv = out.groupby(['kind', 'scheme']).size().unstack(0).fillna(0).astype(int)
    log(piv.to_string())
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='ratlas reconcile ids')
    ap.add_argument('--v1', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    a = ap.parse_args(argv)
    collect(a.v1, a.dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
