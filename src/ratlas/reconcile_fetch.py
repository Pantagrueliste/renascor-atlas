"""Reconcile stage, part 2: Wikidata lookups (network, run on the Mac as ONE niced process).

    python -m ratlas reconcile fetch --ids ids.tsv --dest DIR [--cache DIR] [--max-queries N] [--phase map|details|all]

1. ``map``      authority id -> QID(s) through the identifier properties (batched VALUES, <= 200 per query) and
                Wikipedia titles -> QID through the Wikidata API (<= 50 titles per call).
2. ``details``  places: P625 coordinates, P31 classes, P17 country, en/fr/de/it labels;
                persons: labels, P569/P570 with precision, P106 occupations, P31 classes.

Every response is cached (``wd.WD``), so a rerun costs nothing. The compiled tables are ``wd_map.tsv``,
``wd_places.tsv`` and ``wd_persons.tsv`` in ``--dest``.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .reconcile_ids import PROPS, back_form, query_forms
from .wikidata import DEFAULT_CACHE, WD, bindings, qid_of

BATCH = 200
LANGS = ('en', 'fr', 'de', 'it')


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _lit(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def read_ids(path: Path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------------------------------------
def map_queries(ids) -> list[tuple[str, list[str], str]]:
    """[(scheme, [normalised ids], sparql)] for every identifier property."""
    by = defaultdict(set)
    for r in ids:
        if r['scheme'] in PROPS:
            by[r['scheme']].add(r['id'])
    out = []
    for scheme in sorted(by):
        for ch in _chunks(sorted(by[scheme]), BATCH):
            vals = []
            for i in ch:
                vals += query_forms(scheme, i)
            q = (f'SELECT ?v ?item WHERE {{ VALUES ?v {{ {" ".join(_lit(v) for v in vals)} }} '
                 f'?item wdt:{PROPS[scheme]} ?v . }}')
            out.append((scheme, ch, q))
    return out


def run_map(wd: WD, ids, log, limit=None) -> dict:
    """-> {(scheme, id): sorted [qid...]} (also wikipedia)."""
    res: dict = defaultdict(set)
    qs = map_queries(ids)
    n = 0
    fails = 0
    for scheme, ch, q in qs:
        if limit is not None and n >= limit:
            break
        n += 1
        try:
            resp = wd.sparql(q)
        except Exception as e:  # noqa: BLE001
            fails += 1
            log(f'  map {scheme} batch failed: {e!r}')
            continue
        for b in bindings(resp):
            qid = qid_of(b.get('item'))
            if qid:
                res[(scheme, back_form(scheme, b['v']))].add(qid)
        if n % 10 == 0:
            log(f'  map {n}/{len(qs)} (net {wd.n_net}, cached {wd.n_cached})')
    # Wikipedia titles
    wp = sorted({r['id'] for r in ids if r['scheme'] == 'wikipedia'})
    bysite = defaultdict(list)
    for t in wp:
        lang, _, title = t.partition(':')
        bysite[lang].append(title)
    for lang, titles in sorted(bysite.items()):
        site = f'{lang}wiki'
        for ch in _chunks(sorted(set(titles)), 50):
            if limit is not None and n >= limit:
                break
            n += 1
            try:
                resp = wd.api({'action': 'wbgetentities', 'sites': site, 'titles': '|'.join((t[:1].upper() + t[1:]) for t in ch),
                               'props': 'sitelinks', 'sitefilter': site})
            except Exception as e:  # noqa: BLE001
                fails += 1
                log(f'  wikipedia {site} batch failed: {e!r}')
                continue
            back = {}
            for t in ch:
                back[(t[:1].upper() + t[1:]).replace('_', ' ')] = t
            for qid, ent in (resp.get('entities') or {}).items():
                if not qid.startswith('Q'):
                    continue
                sl = (ent.get('sitelinks') or {}).get(site)
                if sl and sl.get('title') in back:
                    res[('wikipedia', f'{lang}:{back[sl["title"]]}')].add(qid)
    log(f'map done: {len(res)} ids mapped, {fails} failed batches, net {wd.n_net}, cached {wd.n_cached}')
    return {k: sorted(v, key=lambda q: int(q[1:])) for k, v in res.items()}


# ----------------------------------------------------------------------------------------------------------------------
# details through the Wikidata API (wbgetentities, 50 ids per call). The first version used WDQS for this and kept
# timing out at 60 s, so only the identifier mapping uses SPARQL.
# ----------------------------------------------------------------------------------------------------------------------
API_BATCH = 50


def _claims(ent: dict, prop: str):
    """Main-snak datavalues of the non-deprecated claims of ``prop``, preferred rank first."""
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


def _ids(vals):
    return [v['id'] for v in vals if isinstance(v, dict) and 'id' in v]


def _time(vals):
    for v in vals:
        if isinstance(v, dict) and 'time' in v:
            return [v['time'].lstrip('+'), v.get('precision')]
    return None


def reduce_entities(resp: dict) -> dict:
    """wbgetentities response -> {requested id: compact dict}. Redirects are followed by the API."""
    out = {}
    for k, e in (resp.get('entities') or {}).items():
        if 'missing' in e:
            out[k] = {'missing': True}
            continue
        labels = {l: v['value'] for l, v in (e.get('labels') or {}).items()}
        coords = [[v['latitude'], v['longitude']] for v in _claims(e, 'P625') if isinstance(v, dict) and 'latitude' in v]
        iso = [v for v in _claims(e, 'P297') if isinstance(v, str)]
        out[k] = {'id': e.get('id', k), 'labels': labels, 'coords': coords, 'p31': _ids(_claims(e, 'P31')),
                  'p17': _ids(_claims(e, 'P17')), 'p106': _ids(_claims(e, 'P106')),
                  'birth': _time(_claims(e, 'P569')), 'death': _time(_claims(e, 'P570')), 'iso': iso[:1]}
    return out


def fetch_entities(wd: WD, qids, log, props='labels|claims', langs='en|fr|de|it', limit=None):
    got: dict = {}
    qids = sorted(set(qids), key=lambda q: int(q[1:]))
    chunks = list(_chunks(qids, API_BATCH))
    for i, ch in enumerate(chunks):
        if limit is not None and i >= limit:
            break
        try:
            resp = wd.api({'action': 'wbgetentities', 'ids': '|'.join(ch), 'props': props, 'languages': langs},
                          reduce=reduce_entities)
        except Exception as e:  # noqa: BLE001
            log(f'  wbgetentities batch failed: {e!r}')
            continue
        got.update(resp)
        if (i + 1) % 10 == 0:
            log(f'  entities {i + 1}/{len(chunks)} batches (net {wd.n_net}, cached {wd.n_cached})')
    return got


def run_details(wd: WD, place_qids, person_qids, log, limit=None):
    plr = fetch_entities(wd, place_qids, log, limit=limit)
    log(f'places fetched: {len(plr)}')
    per = fetch_entities(wd, person_qids, log, limit=limit)
    log(f'persons fetched: {len(per)}')
    # labels of classes, countries, occupations (English)
    want = set()
    for d in plr.values():
        want.update(d.get('p31', []))
        want.update(d.get('p17', [])[:1])
    for d in per.values():
        want.update(d.get('p31', []))
        want.update(d.get('p106', []))
    countries = {d['p17'][0] for d in plr.values() if d.get('p17')}
    lab = fetch_entities(wd, want - countries, log, props='labels', langs='en', limit=limit)
    lab.update(fetch_entities(wd, countries, log, props='labels|claims', langs='en', limit=limit))
    log(f'labels fetched: {len(lab)}')

    def L(q):
        d = lab.get(q) or {}
        return (d.get('labels') or {}).get('en', '')

    places: dict = {}
    for q, d in plr.items():
        if d.get('missing'):
            continue
        co = d['p17'][0] if d.get('p17') else None
        places[q] = {'coords': {tuple(c) for c in d['coords']}, 'labels': d['labels'],
                     'classes': {c: L(c) for c in d['p31']},
                     'country': (co, L(co), (lab.get(co) or {}).get('iso', [''])[0] if (lab.get(co) or {}).get('iso') else '') if co else None,
                     'redirect_to': d['id'] if d['id'] != q else None}
    persons: dict = {}
    for q, d in per.items():
        if d.get('missing'):
            continue
        persons[q] = {'labels': d['labels'], 'birth': tuple(d['birth']) if d['birth'] else None,
                      'death': tuple(d['death']) if d['death'] else None,
                      'occ': {o: L(o) for o in d['p106']}, 'classes': {c: L(c) for c in d['p31']}}
    log(f'details done: {len(places)} places, {len(persons)} persons with data')
    return places, persons


# ----------------------------------------------------------------------------------------------------------------------
def write_tables(dest: Path, mapping: dict, places: dict, persons: dict):
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / 'wd_map.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['scheme', 'id', 'qids'])
        for (s, i), qs in sorted(mapping.items()):
            w.writerow([s, i, '|'.join(qs)])
    with open(dest / 'wd_places.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid', 'lat', 'lon', 'n_coords', 'label_en', 'label_fr', 'label_de', 'label_it', 'classes',
                    'country_qid', 'country', 'country_iso'])
        for q in sorted(places, key=lambda x: int(x[1:])):
            d = places[q]
            co = sorted(d['coords'])
            lat, lon = co[0] if co else ('', '')
            c = d['country'] or ('', '', '')
            w.writerow([q, lat, lon, len(co)] + [d['labels'].get(l, '') for l in LANGS]
                       + ['|'.join(f'{k}:{v}' for k, v in sorted(d['classes'].items(), key=lambda kv: int(kv[0][1:])))]
                       + list(c))
    freq = Counter()
    for d in persons.values():
        freq.update(d['occ'].keys())
    with open(dest / 'wd_persons.tsv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid', 'label_en', 'label_fr', 'label_de', 'label_it', 'birth', 'birth_prec', 'death', 'death_prec',
                    'occupations', 'n_occupations', 'classes'])
        for q in sorted(persons, key=lambda x: int(x[1:])):
            d = persons[q]
            occ = sorted(d['occ'].items(), key=lambda kv: (-freq[kv[0]], int(kv[0][1:])))[:3]
            b = d['birth'] or ('', '')
            de = d['death'] or ('', '')
            w.writerow([q] + [d['labels'].get(l, '') for l in LANGS] + [b[0], b[1], de[0], de[1]]
                       + ['|'.join(f'{k}:{v}' for k, v in occ), len(d['occ']),
                          '|'.join(f'{k}:{v}' for k, v in sorted(d['classes'].items(), key=lambda kv: int(kv[0][1:])))])


def explicit_qids(ids):
    return {(r['kind'], r['id']) for r in ids if r['scheme'] == 'wikidata'}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='ratlas reconcile fetch')
    ap.add_argument('--ids', type=Path, required=True)
    ap.add_argument('--dest', type=Path, required=True)
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--max-queries', type=int, default=None, help='stop after N queries per phase (tests)')
    ap.add_argument('--phase', choices=['map', 'details', 'all'], default='all')
    ap.add_argument('--offline', action='store_true', help='use the cache only (missing batches count as failed)')
    a = ap.parse_args(argv)
    t0 = time.time()

    def log(m):
        print(f'[{time.time() - t0:6.0f}s] {m}', flush=True)

    ids = read_ids(a.ids)
    wd = WD(a.cache, offline=a.offline, log=log)
    log(f'{len(ids)} ids; cache {a.cache}')
    mapping = run_map(wd, ids, log, a.max_queries)
    if a.phase == 'map':
        write_tables(a.dest, mapping, {}, {})
        return 0
    place_q, person_q = set(), set()
    for r in ids:
        if r['scheme'] == 'wikidata':
            (place_q if r['kind'] == 'place' else person_q).add(r['id'])
        elif (r['scheme'], r['id']) in mapping:
            if r['kind'] == 'person' and r['scheme'] in ('geonames', 'tgn', 'pleiades'):
                continue          # place authorities on a person row (birth place...) do not identify the person
            (place_q if r['kind'] == 'place' else person_q).update(mapping[(r['scheme'], r['id'])])
    log(f'QIDs to describe: {len(place_q)} places, {len(person_q)} persons')
    places, persons = run_details(wd, place_q, person_q, log, a.max_queries)
    write_tables(a.dest, mapping, places, persons)
    log(f'tables written to {a.dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
