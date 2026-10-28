"""Authority labels for the name gate (network, run on the Mac as ONE niced process; v2 reconcile).

    python -m ratlas authority fetch --ids ids.tsv --old-rdir DIR --dest DIR [--cache DIR] [--max-calls N]
    python -m ratlas authority compile --dest DIR           (offline: rebuilds the tables from the cache)

What it fetches (every response cached, so a rerun costs nothing and works offline):

1. ``map``      authority id -> QID for the ids that are new since the v1 reconcile (``--old-rdir``): the Wikidata Action API
                (``list=search`` with ``haswbstatement:P227=a|P227=b``, 25 ids per call), Wikipedia titles through
                ``wbgetentities?sites=&titles=``. The SPARQL service (query.wikidata.org) is NOT used: its robots.txt
                disallows /sparql.
2. ``lobid``    lobid-GND (https://lobid.org/gnd/search, CC0): preferred / variant names, types and Wikidata ``sameAs`` of the GND
                ids that no QID carries, and of every GND id of a place (the GND type says whether it is a person).
                robots.txt of lobid.org only disallows the old 1.x paths.
3. ``details``  ``wbgetentities`` (50 ids per call): labels and aliases in ten languages (en fr de it es la nl pt pl cs, plus
                ``mul``), P31 classes, P2959 (permanent duplicate -> the real item), redirects (merge targets), birth / death,
                P625 coordinates, P17, the identifier statements (to turn ids into QIDs) and the number of statements.

NOT fetched: Getty TGN. ``vocab.getty.edu/robots.txt`` is ``User-agent: * / Disallow: /`` (checked 2026-10-17), so neither the SPARQL
endpoint nor the dumps are touched. TGN ids that Wikidata maps (P1667) are checked through the labels of the item; the others stay
unverified ('suggested').

Rules: User-Agent ``Renascor-Atlas/0.1 (research; https://pantagrueliste.github.io/renascor)``, one request per second at most.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .reconcile_ids import PROPS, back_form, norm_id, query_forms
from .wikidata import DEFAULT_CACHE, UA, WD, qid_of

LANGS10 = ('en', 'fr', 'de', 'it', 'es', 'la', 'nl', 'pt', 'pl', 'cs')
LABEL_LANGS = LANGS10 + ('mul',)
IDENT_PROP = {v: k for k, v in PROPS.items()}          # P227 -> gnd
SEARCH_BATCH = 25
ENT_BATCH = 50
LOBID_BATCH = 50
LOBID = 'https://lobid.org/gnd/search'
DUP_PROPS = ('P2959',)                                   # permanent duplicated item


# ----------------------------------------------------------------------------------------------------------------------
class Net(WD):
    """WD client plus a cached GET for lobid-GND (same politeness)."""

    def get_json(self, url: str, kind: str = 'lobid') -> dict:
        p = self._path(kind, url)
        got = self._get_cached(p)
        if got is not None:
            return got
        if self.offline:
            raise RuntimeError('offline and not cached')
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
        resp = self._fetch(req)
        self._put(p, url, resp)
        return resp


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------------------------------------------------------------
def _claims(ent: dict, prop: str):
    out = []
    for c in (ent.get('claims') or {}).get(prop, []):
        rank = c.get('rank', 'normal')
        if rank == 'deprecated':
            continue
        ms = c.get('mainsnak') or {}
        dv = ms.get('datavalue')
        if ms.get('snaktype') != 'value' or not dv:
            continue
        out.append((0 if rank == 'preferred' else 1, dv.get('value')))
    out.sort(key=lambda t: t[0])
    return [v for _, v in out]


def reduce_gate(resp: dict) -> dict:
    """wbgetentities response -> {requested id: compact dict}; keeps only what the gate needs."""
    out = {}
    for k, e in (resp.get('entities') or {}).items():
        if 'missing' in e:
            out[k] = {'missing': True}
            continue
        labels = {l: v['value'] for l, v in (e.get('labels') or {}).items()}
        aliases = {l: [a['value'] for a in v] for l, v in (e.get('aliases') or {}).items()}
        cl = e.get('claims') or {}
        coords = [[v['latitude'], v['longitude']] for v in _claims(e, 'P625') if isinstance(v, dict) and 'latitude' in v]

        def ids_of(prop):
            return [v['id'] for v in _claims(e, prop) if isinstance(v, dict) and 'id' in v]

        def time_of(prop):
            for v in _claims(e, prop):
                if isinstance(v, dict) and 'time' in v:
                    return [v['time'].lstrip('+'), v.get('precision')]
            return None
        idents = {}
        for prop in IDENT_PROP:
            vs = [v for v in _claims(e, prop) if isinstance(v, str)]
            if vs:
                idents[prop] = vs
        iso = [v for v in _claims(e, 'P297') if isinstance(v, str)]
        out[k] = {'id': e.get('id', k), 'labels': labels, 'aliases': aliases, 'coords': coords, 'p31': ids_of('P31'),
                  'p17': ids_of('P17'), 'p106': ids_of('P106'), 'dup': ids_of('P2959'), 'birth': time_of('P569'), 'death': time_of('P570'),
                  'idents': idents, 'iso': iso[:1], 'n_claims': int(sum(len(v) for v in cl.values()))}
    return out


def reduce_search(resp: dict) -> dict:
    s = (resp.get('query') or {}).get('search') or []
    return {'titles': [x['title'] for x in s if str(x.get('title', '')).startswith('Q')],
            'total': int(((resp.get('query') or {}).get('searchinfo') or {}).get('totalhits', len(s)))}


# ----------------------------------------------------------------------------------------------------------------------
def search_queries(wanted: dict) -> list:
    """{scheme: [ids]} -> [(scheme, [ids], srsearch)]"""
    out = []
    for scheme in sorted(wanted):
        for ch in _chunks(sorted(wanted[scheme]), SEARCH_BATCH):
            vals = []
            for i in ch:
                vals += query_forms(scheme, i)
            q = 'haswbstatement:' + '|'.join(f'{PROPS[scheme]}={v}' for v in vals)
            out.append((scheme, ch, q))
    return out


def run_search(wd: Net, wanted: dict, log, limit=None) -> set:
    qids: set = set()
    qs = search_queries(wanted)
    n = fails = 0
    for scheme, ch, q in qs:
        if limit is not None and n >= limit:
            break
        n += 1
        off = 0
        while True:
            try:
                r = wd.api({'action': 'query', 'list': 'search', 'srsearch': q, 'srprop': '', 'srlimit': '50',
                            'sroffset': str(off)}, reduce=reduce_search)
            except Exception as e:  # noqa: BLE001
                fails += 1
                log(f'  search {scheme} batch failed: {e!r}')
                break
            qids.update(r['titles'])
            off += 50
            if off >= r['total'] or off >= 500 or not r['titles']:
                break
        if n % 25 == 0:
            log(f'  search {n}/{len(qs)} (net {wd.n_net}, cached {wd.n_cached}, {len(qids)} QIDs)')
    log(f'search done: {len(qids)} QIDs, {fails} failed batches')
    return qids


def fetch_gate(wd: Net, qids: Iterable[str], log, limit=None) -> dict:
    got: dict = {}
    qs = sorted(set(qids), key=lambda q: int(q[1:]))
    chunks = list(_chunks(qs, ENT_BATCH))
    for i, ch in enumerate(chunks):
        if limit is not None and i >= limit:
            break
        try:
            resp = wd.api({'action': 'wbgetentities', 'ids': '|'.join(ch), 'props': 'labels|aliases|claims',
                           'languages': '|'.join(LABEL_LANGS)}, reduce=reduce_gate)
        except Exception as e:  # noqa: BLE001
            log(f'  wbgetentities batch failed: {e!r}')
            continue
        got.update(resp)
        if (i + 1) % 20 == 0:
            log(f'  entities {i + 1}/{len(chunks)} batches (net {wd.n_net}, cached {wd.n_cached})')
    log(f'entities: {len(got)} fetched')
    return got


def fetch_class_labels(wd: Net, qids: Iterable[str], log, limit=None) -> dict:
    out: dict = {}
    for ch in _chunks(sorted(set(qids), key=lambda q: int(q[1:])), ENT_BATCH):
        try:
            resp = wd.api({'action': 'wbgetentities', 'ids': '|'.join(ch), 'props': 'labels', 'languages': 'en'},
                          reduce=lambda r: {k: ((e.get('labels') or {}).get('en') or {}).get('value', '')
                                            for k, e in (r.get('entities') or {}).items()})
        except Exception as e:  # noqa: BLE001
            log(f'  class labels batch failed: {e!r}')
            continue
        out.update(resp)
    return out


def idents_to_map(ents: dict, wanted_norm: dict) -> dict:
    """identifier statements of fetched items -> {(scheme, id): {qid}} for the ids we asked about.
    ``wanted_norm`` {scheme: set(normalised ids)}"""
    res: dict = defaultdict(set)
    for q, d in ents.items():
        if d.get('missing'):
            continue
        real = d.get('id', q)
        for prop, vals in (d.get('idents') or {}).items():
            scheme = IDENT_PROP[prop]
            ws = wanted_norm.get(scheme)
            if not ws:
                continue
            for v in vals:
                nid = back_form(scheme, v)
                if nid in ws:
                    res[(scheme, nid)].add(real)
    return res


# ----------------------------------------------------------------------------------------------------------------------
def lobid_records(wd: Net, gnd_ids: Iterable[str], log, limit=None) -> dict:
    out: dict = {}
    chunks = list(_chunks(sorted(set(gnd_ids)), LOBID_BATCH))
    for i, ch in enumerate(chunks):
        if limit is not None and i >= limit:
            break
        q = 'gndIdentifier:(' + ' OR '.join(ch) + ')'
        url = LOBID + '?' + urllib.parse.urlencode({'q': q, 'format': 'json', 'size': str(len(ch) + 10)})
        try:
            r = wd.get_json(url)
        except Exception as e:  # noqa: BLE001
            log(f'  lobid batch failed: {e!r}')
            continue
        for m in r.get('member', []):
            gid = m.get('gndIdentifier')
            if not gid:
                continue
            wdq = [s['id'].rsplit('/', 1)[-1] for s in (m.get('sameAs') or []) if 'wikidata.org/entity/' in s.get('id', '')]
            out[gid] = {
                'name': m.get('preferredName'), 'variants': (m.get('variantName') or [])[:60],
                'types': [t for t in (m.get('type') or []) if t not in ('AuthorityResource',)],
                'birth': (m.get('dateOfBirth') or [None])[0], 'death': (m.get('dateOfDeath') or [None])[0],
                'qids': wdq, 'info': (m.get('biographicalOrHistoricalInformation') or [None])[0]}
        if (i + 1) % 20 == 0:
            log(f'  lobid {i + 1}/{len(chunks)} (net {wd.n_net}, cached {wd.n_cached}, {len(out)} records)')
    log(f'lobid: {len(out)} records')
    return out


# ----------------------------------------------------------------------------------------------------------------------
def _read_ids(path: Path) -> list:
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def _read_map(path: Path) -> dict:
    out: dict = defaultdict(set)
    if not Path(path).exists():
        return out
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            for q in r['qids'].split('|'):
                if q:
                    out[(r['scheme'], r['id'])].add(q)
    return out


def gnd_of(scheme: str, idv: str) -> Optional[str]:
    """GND id behind a 'gnd' or a 'deutsche-biographie' id."""
    if scheme == 'gnd':
        return idv
    if scheme == 'deutsche-biographie':
        return idv
    return None


def write_outputs(dest: Path, mapping: dict, ents: dict, classes: dict, lobid: dict, log):
    import pandas as pd
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'wd_map.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['scheme', 'id', 'qids'])
        for (s, i), qs in sorted(mapping.items()):
            w.writerow([s, i, '|'.join(sorted(qs, key=lambda q: int(q[1:])))])
    rows = []
    for q, d in ents.items():
        if d.get('missing'):
            rows.append({'qid': q, 'missing': True})
            continue
        co = sorted(tuple(c) for c in d['coords'])
        rows.append({
            'qid': q, 'missing': False, 'redirect_to': d['id'] if d['id'] != q else None, 'dup_of': d['dup'] or None,
            'classes': d['p31'], 'occupations': d.get('p106', [])[:12], 'labels': json.dumps(d['labels'], ensure_ascii=False),
            'aliases': json.dumps(d['aliases'], ensure_ascii=False),
            'lat': co[0][0] if co else None, 'lon': co[0][1] if co else None, 'n_coords': len(co),
            'country_qid': d['p17'][0] if d['p17'] else None, 'country_iso': d['iso'][0] if d['iso'] else None,
            'birth': d['birth'][0] if d['birth'] else None, 'birth_prec': d['birth'][1] if d['birth'] else None,
            'death': d['death'][0] if d['death'] else None, 'death_prec': d['death'][1] if d['death'] else None,
            'n_claims': d['n_claims']})
    df = pd.DataFrame(rows)
    df['n_coords'] = df['n_coords'].fillna(0).astype('int64')
    df['n_claims'] = df['n_claims'].fillna(0).astype('int64')
    df['classes'] = df['classes'].map(lambda v: v if isinstance(v, list) else [])
    df['occupations'] = df['occupations'].map(lambda v: v if isinstance(v, list) else [])
    df.to_parquet(dest / 'wd_entities.parquet', index=False)
    with open(dest / 'wd_class_labels.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid', 'label_en'])
        for q in sorted(classes, key=lambda x: int(x[1:])):
            w.writerow([q, classes[q]])
    lr = [{'gnd': g, 'name': d['name'], 'variants': d['variants'], 'types': d['types'], 'birth': d['birth'],
           'death': d['death'], 'qids': d['qids'], 'info': d['info']} for g, d in lobid.items()]
    pd.DataFrame(lr, columns=['gnd', 'name', 'variants', 'types', 'birth', 'death', 'qids', 'info']).to_parquet(
        dest / 'gnd_records.parquet', index=False)
    log(f'tables written to {dest}: {len(mapping)} mapped ids, {len(df)} items, {len(classes)} class labels, {len(lr)} GND records')


def run(ids_path: Path, old_rdir: Path, dest: Path, cache: Path, limit: Optional[int] = None, offline: bool = False, log=print):
    t0 = time.time()
    wd = Net(cache, offline=offline, log=log)
    ids = _read_ids(ids_path)
    old_ids = {(r['scheme'], r['id']) for r in _read_ids(old_rdir / 'ids.tsv')}
    mapping = _read_map(old_rdir / 'wd_map.tsv')
    log(f'{len(ids)} ids; {len(old_ids)} known from v1; {len(mapping)} already mapped')
    # ---- 1. new ids with an identifier property
    wanted: dict = defaultdict(set)
    wanted_all: dict = defaultdict(set)
    for r in ids:
        s, i = r['scheme'], r['id']
        if s in PROPS:
            wanted_all[s].add(i)
            if (s, i) not in old_ids and (s, i) not in mapping:
                wanted[s].add(i)
    log('new ids to look up: ' + ', '.join(f'{s} {len(v)}' for s, v in sorted(wanted.items())))
    q_search = run_search(wd, wanted, log, limit)
    # new Wikipedia titles
    new_wp = [r for r in ids if r['scheme'] == 'wikipedia' and ('wikipedia', r['id']) not in old_ids
              and ('wikipedia', r['id']) not in mapping]
    if new_wp:
        from .reconcile_fetch import run_map
        wmap = run_map(wd, new_wp, log, limit)
        for k, v in wmap.items():
            if k[0] == 'wikipedia':
                mapping[k].update(v)
    # ---- 2. lobid-GND for the GND ids without a QID, and for every GND id of a place
    # the GND ids carried by Wikidata items found so far are known only after the details fetch; ask lobid for all ids that no
    # (old or new) mapping covers yet
    ents = fetch_gate(wd, q_search, log, limit)
    wanted_norm = {s: set(v) for s, v in wanted_all.items()}
    for k, v in idents_to_map(ents, wanted_norm).items():
        mapping[k].update(v)
    place_gnd = {r['id'] for r in ids if r['scheme'] in ('gnd', 'deutsche-biographie') and r['kind'] == 'place'}
    gnd_need = set()
    for r in ids:
        if r['scheme'] in ('gnd', 'deutsche-biographie'):
            if (r['scheme'], r['id']) not in mapping or r['id'] in place_gnd:
                gnd_need.add(r['id'])
    lobid = lobid_records(wd, gnd_need, log, limit)
    schemes_of: dict = defaultdict(set)
    for r in ids:
        if r['scheme'] in ('gnd', 'deutsche-biographie'):
            schemes_of[r['id']].add(r['scheme'])
    for g, d in lobid.items():
        for q in d['qids']:
            for s in schemes_of.get(g, ()):
                if (s, g) not in mapping or not mapping[(s, g)]:
                    mapping[(s, g)].add(q)
    # ---- 3. details of every candidate QID: mapped, explicit, redirect / duplicate targets
    cand = {q for v in mapping.values() for q in v}
    cand |= {r['id'] for r in ids if r['scheme'] == 'wikidata'}
    ents.update(fetch_gate(wd, cand - set(ents), log, limit))
    for k, v in idents_to_map({q: d for q, d in ents.items() if q in cand}, wanted_norm).items():
        mapping[k].update(v)
    extra = set()
    for d in ents.values():
        if not d.get('missing'):
            extra.update(d['dup'])
            if d['id'] in ents or d['id'].startswith('Q'):
                extra.add(d['id'])
    ents.update(fetch_gate(wd, extra - set(ents), log, limit))
    clsq = {c for d in ents.values() if not d.get('missing') for c in d['p31'] + d.get('p106', [])[:12] + d['p17'][:1]}
    classes = fetch_class_labels(wd, clsq, log, limit)
    write_outputs(dest, mapping, ents, classes, lobid, log)
    log(f'done in {time.time() - t0:.0f}s (net {wd.n_net}, cached {wd.n_cached})')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas authority')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('fetch')
    p.add_argument('--ids', type=Path, required=True)
    p.add_argument('--old-rdir', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--max-calls', type=int, default=None)
    p.add_argument('--offline', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()

    def log(m):
        print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)
    run(a.ids, a.old_rdir, a.dest, a.cache, a.max_calls, a.offline, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
