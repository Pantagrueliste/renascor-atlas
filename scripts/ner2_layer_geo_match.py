"""NER layer v1, geocoding stage 4 (Mac, nice): match the layer's points with the editor-tagged points of the atlas v2.4 and write ``model_layer.json``.

    nice -n 19 python scripts/ner2_layer_geo_match.py --points points.json --globe globe_places.json --out model_layer.json [--summary summary.json]

A layer point coincides with an editor-tagged point when they carry the same Wikidata id, or when they lie less than 30 km apart (two areas: less than the smaller of their two radii, when that is more than 30 km) and share a name (the
point's name or one of its written forms against the globe point's name, compared as geocoder keys: case, accents, long s, Latin and German endings and early-modern spelling
skeletons). Each layer point gets ``ed`` = the nearest coinciding editor point ({id, name, km, how}) or nothing.

``model_layer.json``: ``header`` and ``places`` in the row shape of the v2.4 globe (id, name, lat, lon, k, n, q, c, cols, s, r) with the layer's own source code
``s = "m"`` and, per point: nd (documents), gs (method of the coordinates), conf, qid, gid (GeoNames id), forms ([form, mentions, language]), nforms, langs, docs (top documents),
ed (coinciding editor-tagged point).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ratlas import geotext as GT
from ratlas import polities as POL
from ratlas import exonyms as EX

QUARTERS = list(range(1450, 1700, 25))


def keys_of(s: str) -> set:
    """Keys of a name; of 'Madrid, Valdemoro' only the leaf (the last part, or the first when the last is a qualifier: 'England, Königreich')."""
    parts = [x.strip() for x in str(s).split(',') if x.strip()]
    if len(parts) > 1:
        s = parts[0] if POL.is_qualifier(GT.forms(parts[-1]).keys(('t0',))[0] if GT.forms(parts[-1]).keys(('t0',)) else '') else parts[-1]
    f = GT.forms(s)
    out = set()
    for k in f.keys(('t0', 't1', 'paren')):
        out.add(k)
        out.add(GT.skeleton(k))
    for k in list(out):                    # the curated exonym table links the foreign names of a place (Agypten = Egypt, Florenz = Florence)
        try:
            h = EX.lookup(k)
        except Exception:
            h = None
        if h:
            for lk in GT.forms(h[3]).keys(('t0',)):
                out.add('ex:' + lk)
    return {k for k in out if len(k) >= 3}


def hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype='float64')) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--globe', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--summary', type=Path, default=None)
    ap.add_argument('--stats', type=Path, default=None, help='summary.json of the points step (fates), copied into the header')
    ap.add_argument('--km', type=float, default=30.0)
    a = ap.parse_args()
    t0 = time.time()
    pts = json.loads(a.points.read_text('utf-8'))
    G = json.loads(a.globe.read_text('utf-8'))['places']
    glat = np.array([g['lat'] for g in G])
    glon = np.array([g['lon'] for g in G])
    gkeys = [keys_of(g['name']) for g in G]
    gr = np.array([float(g.get('r') or 0) for g in G])
    by_qid = {}
    for i, g in enumerate(G):
        if g.get('qid'):
            by_qid.setdefault(g['qid'], []).append(i)
    n_match = 0
    m_match = 0
    editor_hit = set()
    how_c = Counter()
    for p in pts:
        d = hav(p['lat'], p['lon'], glat, glon)
        r_l = float(p.get('r') or 0)
        thr = np.where((r_l > 0) & (gr > 0), np.maximum(a.km, np.minimum(r_l, gr)), a.km)
        near = np.flatnonzero(d <= thr)
        pk = keys_of(p['name'])
        for f in p['forms']:
            pk |= keys_of(f[0])
        cands = []
        if p.get('qid') and p['qid'] in by_qid:
            cands += [(int(i), 'wikidata') for i in by_qid[p['qid']]]
        cands += [(int(i), 'name') for i in near if gkeys[i] & pk]
        best = None
        if cands:
            # the editor point with the most mentions among those that coincide (wikidata first)
            best = max(cands, key=lambda c: (c[1] == 'wikidata', G[c[0]]['n']))
            p['ed_n'] = len(cands)
        if best is not None:
            i, how = best
            g = G[i]
            p['ed'] = {'id': g['id'], 'name': g['name'], 'km': round(float(d[i]), 1), 'how': how, 'n': g['n']}
            if p['k'] == 'a' and g.get('r') and g['r'] > r_l:          # an area drawn with a point radius of 5 km: the editors' radius for the same area
                p['r0'] = int(r_l)
                p['r'] = int(g['r'])
            n_match += 1
            m_match += p['n']
            editor_hit.add(i)
            how_c[how] += 1
    tot_m = sum(p['n'] for p in pts)
    hdr = {'title': 'Renascor Atlas, places found by the NER model on untagged texts (layer m)', 'generated': time.strftime('%Y-%m-%d'),
           'period': [1450, 1700], 'quarters': QUARTERS,
           'fields': {'n': 'mentions (spans with confidence >= 0.98 and >= 2 of 4 models agreeing)', 'nd': 'documents', 'q': 'mentions per quarter-century by document year', 'k': 'kind c/a/n/b/u as in the globe',
                      'r': 'for areas: radius in km', 's': '"m" = found by the model (never an editor-tagged place)',
                      'gs': 'method of the coordinates: name_exonym, name_latin, name_geonames, name_propagation', 'conf': 'high / medium (geocoder confidence)',
                      'forms': '[written form, mentions, language] (the forms of the texts, cleaned)', 'docs': 'top documents: id, title (first line of the text), year, collection, url (built from the identifier, not checked), n',
                      'ed': 'the editor-tagged point of v2.4 that coincides (same Wikidata id, or < 30 km and same name)'},
           'places': len(pts), 'mentions': tot_m, 'coincide_with_editor_points': {'points': n_match, 'mentions': m_match, 'by': dict(how_c), 'editor_points_hit': len(editor_hit)},
           'note': 'predictions of the model, no human review; a separate layer, distinct from the editors\' places'}
    if a.stats and a.stats.exists():
        hdr['fates'] = json.loads(a.stats.read_text('utf-8'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'header': hdr, 'places': pts}, ensure_ascii=False, separators=(',', ':')), 'utf-8')
    summ = {'points': len(pts), 'mentions': tot_m, 'coincide_points': n_match, 'coincide_mentions': m_match, 'by': dict(how_c), 'editor_points_hit': len(editor_hit),
            'only_model_points': len(pts) - n_match, 'only_model_mentions': tot_m - m_match, 'bytes': a.out.stat().st_size, 'seconds': round(time.time() - t0)}
    if a.summary:
        a.summary.write_text(json.dumps(summ, indent=1), 'utf-8')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
