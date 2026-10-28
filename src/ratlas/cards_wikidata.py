"""Place cards, part 1: Wikidata details for every place QID (network; run on Hypatia through Slurm).

    python -m ratlas.cards_wikidata --v1 DIR --dest DIR [--cache DIR] [--max-calls N]

Reads ``places_global.parquet`` and ``places_local_geo.parquet`` of ``--v1`` (QIDs of both), asks the Wikidata API
(``wbgetentities``, 50 ids per call, one request per second, every response cached by ``wikidata.WD``) and writes
``wd_places.parquet`` into ``--dest``.

Pass 1: labels / descriptions / aliases in fr en de it la es nl, sitelinks (frwiki enwiki dewiki itwiki) and claims
(P31 P17 P131 P625 P1082 P571 P18 P1566 P1667 P227 P1584), claims shrunk BEFORE caching.
Pass 2: fr/en labels of the items referenced by P31, P17 and P131.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

from .wikidata import DEFAULT_CACHE, WD

LANGS = ('fr', 'en', 'de', 'it', 'la', 'es', 'nl')
WIKIS = ('fr', 'en', 'de', 'it')
CLAIMS = ('P31', 'P17', 'P131', 'P625', 'P1082', 'P571', 'P18', 'P1566', 'P1667', 'P227', 'P1584')
BATCH = 50


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------------------------------------------------------------
def _snak(sn):
    """Simplified value of a snak, None for novalue/somevalue."""
    if not sn or sn.get('snaktype') != 'value':
        return None
    dv = sn.get('datavalue') or {}
    t, v = dv.get('type'), dv.get('value')
    if t == 'wikibase-entityid':
        return v.get('id')
    if t == 'globecoordinate':
        return [v.get('latitude'), v.get('longitude')]
    if t == 'quantity':
        return v.get('amount')
    if t == 'time':
        return {'t': v.get('time'), 'p': v.get('precision')}
    return v  # string, external-id, commonsMedia


def reduce_entities(resp: dict) -> dict:
    """Shrink a wbgetentities response before caching."""
    out = {'entities': {}}
    for key, e in (resp.get('entities') or {}).items():
        if 'missing' in e:
            out['entities'][key] = {'id': e.get('id', key), 'missing': True}
            continue
        r = {'id': e.get('id', key)}
        for f in ('labels', 'descriptions'):
            if e.get(f):
                r[f] = {k: v['value'] for k, v in e[f].items()}
        if e.get('aliases'):
            r['aliases'] = {k: [a['value'] for a in v] for k, v in e['aliases'].items()}
        if e.get('sitelinks'):
            r['sitelinks'] = {k: v.get('title') for k, v in e['sitelinks'].items()}
        cl = {}
        for p in CLAIMS:
            lst = []
            for c in (e.get('claims') or {}).get(p, []):
                if c.get('rank') == 'deprecated':
                    continue
                v = _snak(c.get('mainsnak'))
                if v is None:
                    continue
                item = {'v': v, 'r': c.get('rank', 'normal')[0]}
                if p == 'P1082':
                    q = [_snak(s) for s in (c.get('qualifiers') or {}).get('P585', [])]
                    q = [x for x in q if isinstance(x, dict)]
                    if q:
                        item['pit'] = q[0]['t']
                lst.append(item)
            if lst:
                cl[p] = lst
        if cl:
            r['claims'] = cl
        out['entities'][key] = r
    if resp.get('redirects'):
        out['redirects'] = resp['redirects']
    return out


def reduce_labels(resp: dict) -> dict:
    out = {'entities': {}}
    for key, e in (resp.get('entities') or {}).items():
        if 'missing' in e:
            out['entities'][key] = {'id': key, 'labels': {}}
            continue
        out['entities'][key] = {'id': e.get('id', key), 'labels': {k: v['value'] for k, v in (e.get('labels') or {}).items()}}
    return out


def _one_pass(wd: WD, chunks, props_params, reduce, ents, log, tag):
    left = []
    for i, ch in enumerate(chunks, 1):
        params = dict(props_params, action='wbgetentities', ids='|'.join(ch))
        resp = wd.api(params, reduce=reduce)
        ents.update(resp['entities'])
        for q in ch:  # the API silently drops entities when the answer is too large: retry those in smaller calls
            if q not in resp['entities']:
                left.append(q)
        if i % 10 == 0 or i == len(chunks):
            log(f'  {tag}: {i}/{len(chunks)} calls (net {wd.n_net}, cached {wd.n_cached}, to retry {len(left)})')
    return left


def fetch_entities(wd: WD, qids, max_calls=None, log=print) -> dict:
    ents = {}
    todo = sorted(qids, key=lambda q: int(q[1:]))
    pp = {'props': 'labels|descriptions|aliases|claims|sitelinks', 'languages': '|'.join(LANGS),
          'sitefilter': '|'.join(w + 'wiki' for w in WIKIS)}
    for size in (BATCH, 10, 3, 1):
        if not todo:
            break
        todo = _one_pass(wd, list(_chunks(todo, size)), pp, reduce_entities, ents, log, f'pass 1 (x{size})')
    for q in todo:
        log(f'  never returned: {q}')
    return ents


def fetch_labels(wd: WD, qids, log=print) -> dict:
    ents = {}
    todo = sorted(qids, key=lambda q: int(q[1:]))
    pp = {'props': 'labels', 'languages': 'fr|en'}
    for size in (BATCH, 10, 3, 1):
        if not todo:
            break
        todo = _one_pass(wd, list(_chunks(todo, size)), pp, reduce_labels, ents, log, f'pass 2 (x{size})')
    return {k: e['labels'] for k, e in ents.items()}


# ----------------------------------------------------------------------------------------------------------------------
def _best(claims, prop):
    """Claims of ``prop`` with the preferred rank if any, else all (order kept)."""
    lst = claims.get(prop, [])
    pref = [c for c in lst if c['r'] == 'p']
    return pref or lst


def _vals(claims, prop):
    out = []
    for c in _best(claims, prop):
        v = c['v']
        if isinstance(v, str) and v not in out:
            out.append(v)
    return out


def _time(v):
    """('1480', 1480, 9) from {'t': '+1480-00-00T00:00:00Z', 'p': 9}; precision <= 7 gives a century/decade text."""
    t, p = v['t'], v['p']
    neg = t.startswith('-')
    body = t[1:]
    y, mo, d = body[:body.index('-')], body.split('-')[1], body.split('-')[2][:2]
    yi = int(y) * (-1 if neg else 1)
    sign = '-' if neg else ''
    if p is None or p >= 11:
        txt = f'{sign}{y}-{mo}-{d}'
    elif p == 10:
        txt = f'{sign}{y}-{mo}'
    elif p == 9:
        txt = f'{sign}{int(y)}'
    elif p == 8:
        txt = f'{sign}{int(y)}s'
    elif p == 7:
        c = (abs(yi) - 1) // 100 + 1 if yi else 0
        txt = ('1er' if c == 1 else f'{c}e') + ' siècle' + (' av. J.-C.' if neg else '')
    else:
        txt = f'{sign}{int(y)}'
    return txt, yi, p


def _pit_year(t):
    if not t:
        return -10 ** 9
    try:
        return _time({'t': t, 'p': 9})[1]
    except Exception:
        return -10 ** 9


def build_rows(ents: dict, labels: dict, nip: dict) -> list[dict]:
    rows = []

    def lab(q, lang):
        return (labels.get(q) or {}).get(lang)

    for q in sorted(nip, key=lambda x: int(x[1:])):
        e = ents.get(q)
        row = {'qid': q, 'n_in_period': nip[q], 'missing': e is None or bool(e.get('missing'))}
        if e is None or e.get('missing'):
            rows.append(row)
            continue
        if e['id'] != q:
            row['redirect_to'] = e['id']
        for lg in LANGS:
            row[f'label_{lg}'] = (e.get('labels') or {}).get(lg)
            row[f'desc_{lg}'] = (e.get('descriptions') or {}).get(lg)
            row[f'aliases_{lg}'] = (e.get('aliases') or {}).get(lg) or []
        cl = e.get('claims') or {}
        for name, prop in (('instance_of', 'P31'), ('country', 'P17'), ('located_in', 'P131')):
            ids = _vals(cl, prop)
            row[name] = ids
            for lg in ('fr', 'en'):
                row[f'{name}_{lg}'] = [lab(i, lg) or lab(i, 'en') or lab(i, 'fr') or i for i in ids]
        coords = [c for c in _best(cl, 'P625') if isinstance(c['v'], list)]
        if coords:
            row['lat'], row['lon'] = coords[0]['v']
        pops = [c for c in cl.get('P1082', []) if isinstance(c['v'], str)]
        if pops:
            # latest by point in time; undated ones rank below dated ones, preferred rank breaks ties, then list order
            def key(ic):
                i, c = ic
                return (_pit_year(c.get('pit')), c.get('pit') or '', c['r'] == 'p', i)
            c = max(enumerate(pops), key=key)[1]
            try:
                row['population'] = int(float(c['v']))
            except ValueError:
                pass
            if c.get('pit'):
                try:
                    row['population_year'] = _time({'t': c['pit'], 'p': 9})[1]
                except Exception:
                    pass
        inc = [c for c in _best(cl, 'P571') if isinstance(c['v'], dict)]
        if inc:
            try:
                row['inception'], row['inception_year'], row['inception_precision'] = _time(inc[0]['v'])
            except Exception:
                pass
        img = _vals(cl, 'P18')
        if img:
            row['image'] = img[0]
        for name, prop in (('geonames', 'P1566'), ('tgn', 'P1667'), ('gnd', 'P227'), ('pleiades', 'P1584')):
            row[name] = _vals(cl, prop)
        for w in WIKIS:
            t = (e.get('sitelinks') or {}).get(w + 'wiki')
            if t:
                row[f'wp_{w}_title'] = t
                row[f'wp_{w}_url'] = f'https://{w}.wikipedia.org/wiki/' + urllib.parse.quote(t.replace(' ', '_'), safe="_(),'!*~:;@$&=+-./")
        rows.append(row)
    return rows


def schema():
    import pyarrow as pa
    ls, s, d, i = pa.list_(pa.string()), pa.string(), pa.float64(), pa.int64()
    f = [('qid', s), ('n_in_period', i), ('missing', pa.bool_()), ('redirect_to', s)]
    for lg in LANGS:
        f += [(f'label_{lg}', s), (f'desc_{lg}', s), (f'aliases_{lg}', ls)]
    for n in ('instance_of', 'country', 'located_in'):
        f += [(n, ls), (f'{n}_fr', ls), (f'{n}_en', ls)]
    f += [('lat', d), ('lon', d), ('population', i), ('population_year', i), ('inception', s), ('inception_year', i),
          ('inception_precision', i), ('image', s)]
    f += [(n, ls) for n in ('geonames', 'tgn', 'gnd', 'pleiades')]
    for w in WIKIS:
        f += [(f'wp_{w}_title', s), (f'wp_{w}_url', s)]
    return pa.schema(f)


def collect_qids(v1: Path):
    import pyarrow.parquet as pq
    g = pq.read_table(v1 / 'places_global.parquet', columns=['qid', 'n_in_period']).to_pylist()
    l = pq.read_table(v1 / 'places_local_geo.parquet', columns=['qid']).to_pylist()
    nip = defaultdict(int)
    for r in g:
        if r['qid']:
            nip[r['qid']] += r['n_in_period'] or 0
    for r in l:
        if r['qid']:
            nip.setdefault(r['qid'], 0)
    return dict(nip)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--v1', required=True, type=Path)
    ap.add_argument('--dest', required=True, type=Path)
    ap.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    ap.add_argument('--max-calls', type=int, default=0)
    a = ap.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq
    a.dest.mkdir(parents=True, exist_ok=True)
    nip = collect_qids(a.v1)
    print(f'{len(nip)} QIDs', flush=True)
    wd = WD(a.cache, min_interval=1.1)
    ents = fetch_entities(wd, nip, a.max_calls)
    ref = set()
    for e in ents.values():
        for p in ('P31', 'P17', 'P131'):
            for c in (e.get('claims') or {}).get(p, []):
                if isinstance(c['v'], str):
                    ref.add(c['v'])
    print(f'{len(ref)} referenced items', flush=True)
    labels = fetch_labels(wd, ref)
    rows = build_rows(ents, labels, nip)
    sch = schema()
    t = pa.Table.from_pylist(rows, schema=sch)
    pq.write_table(t, a.dest / 'wd_places.parquet', compression='zstd')
    n = lambda k: sum(1 for r in rows if r.get(k))
    print(json.dumps({'rows': len(rows), 'missing': n('missing'), 'redirect': n('redirect_to'), 'coords': n('lat'),
                      'image': n('image'), 'population': n('population'), 'inception': n('inception'),
                      'label_fr': n('label_fr'), 'label_en': n('label_en'), 'desc_fr': n('desc_fr'),
                      'desc_en': n('desc_en'), 'wp_fr': n('wp_fr_url'), 'wp_en': n('wp_en_url'),
                      'geonames': n('geonames'), 'tgn': n('tgn'), 'gnd': n('gnd'), 'pleiades': n('pleiades'),
                      'net_calls': wd.n_net, 'cached_calls': wd.n_cached}))


if __name__ == '__main__':
    sys.exit(main())
