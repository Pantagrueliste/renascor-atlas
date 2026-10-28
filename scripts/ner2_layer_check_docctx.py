"""NER layer v1, independent check, step 3 (CPU, Hypatia): how far is a judged mention's point from the other places of its document?

    python scripts/ner2_layer_check_docctx.py --layer ~/renascor-atlas-data/ner/layer_v1 --items DIR/judged_items.json --out DIR/docctx.json

For each judged item (doc_id, pid) writes: number of other placed points in the document, the distance (km) to the nearest other point of >= 50 mentions, the distance to
the nearest other point of any size, and the median distance to the other points. Used to test a "far from its document" rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    ap.add_argument('--items', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    L = Path(a.layer).expanduser()
    items = json.load(open(a.items))
    P = {int(p['id'].split(':')[1]): p for p in json.load(open(L / 'geo/points/points.json'))}
    F = pd.read_parquet(L / 'geo/points/forms_placed.parquet', columns=['lang', 'key', 'pid'])
    F = F[F['pid'] >= 0]
    D = pd.read_parquet(L / 'geo/doc_forms.parquet')
    docs = {it['doc_id'] for it in items}
    D = D[D['doc_id'].isin(docs)].merge(F, on=['lang', 'key'], how='inner')
    by = {k: g.groupby('pid')['n'].sum() for k, g in D.groupby('doc_id')}
    out = []
    for it in items:
        s = by.get(it['doc_id'])
        p = P[it['pid']]
        r = {'doc_id': it['doc_id'], 'pid': it['pid'], 'n_other': 0, 'd_big': None, 'd_any': None, 'd_med': None, 'm_other': 0}
        if s is not None:
            s = s.drop(it['pid'], errors='ignore')
            r['n_other'] = int(len(s))
            r['m_other'] = int(s.sum())
            if len(s):
                pid = s.index.to_numpy()
                lat = np.array([P[i]['lat'] for i in pid])
                lon = np.array([P[i]['lon'] for i in pid])
                nn = np.array([P[i]['n'] for i in pid])
                d = hav(p['lat'], p['lon'], lat, lon)
                r['d_any'] = float(d.min())
                big = nn >= 50
                r['d_big'] = float(d[big].min()) if big.any() else None
                w = s.to_numpy().astype(float)
                o = np.argsort(d)
                cw = np.cumsum(w[o])
                r['d_med'] = float(d[o][np.searchsorted(cw, cw[-1] / 2)])
        out.append(r)
    json.dump(out, open(a.out, 'w'))
    print('done', len(out))


if __name__ == '__main__':
    main()
