"""NER layer v1, independent check, step 2 (CPU, Hypatia): a second draw (mid-sized points, one mention per uniformly drawn small point) and the suspect forms.

    python scripts/ner2_layer_check_extra.py --layer ~/renascor-atlas-data/ner/layer_v1 --out DIR [--seed 11]

Writes DIR/extra.jsonl (60 mentions of points with 11-100 mentions + 40 points with <= 10 mentions, one mention each) and DIR/suspects.jsonl (a few mentions of each
listed form). Same record format as ``ner2_layer_check_sample.py``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_layer_geo_prep import clean_surface, reason  # noqa: E402

SUSPECTS = {'en': ['bath', 'nice', 'reading', 'ware', 'barton', 'iordan', 'jordan', 'hull', 'derby', 'barking', 'megalopolis', 'dan', 'come', 'kent', 'york', 'grove', 'dean', 'barley',
                   'temple', 'warden', 'leek', 'wing', 'over', 'sandy', 'well', 'welling'],
            'it': ['prioli', 'cavalli', 'quer', 'montona', 'istrana', 'terranuova'], 'es': ['rojas', 'toro'], 'fr': ['isle de france', 'nice']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    L = Path(a.layer).expanduser()
    O = Path(a.out).expanduser()
    rng = np.random.default_rng(a.seed)
    s = pq.read_table(L / 'spans.parquet').to_pandas()
    s = s[(s['conf'] >= 0.98) & (s['n_agree'] >= 2)].copy()
    cm = {u: clean_surface(u) for u in s['surface'].unique()}
    s['form'] = s['surface'].map(cm)
    s['why'] = s['surface'].map({u: reason(c) for u, c in cm.items()})
    s = s[s['why'] == ''].copy()
    s['key'] = s['form'].str.casefold()
    F = pd.read_parquet(L / 'geo/points/forms_placed.parquet')
    F = F[F['pid'] >= 0]
    m = s.merge(F[['lang', 'key', 'pid', 'gc_source']], on=['lang', 'key'], how='inner').reset_index(drop=True)
    pts = json.load(open(L / 'geo/points/points.json'))
    P = {int(p['id'].split(':')[1]): p for p in pts}
    m['pn'] = m['pid'].map(lambda i: P[i]['n'])
    mid = m[(m['pn'] > 10) & (m['pn'] <= 100)]
    pick = [mid.loc[rng.choice(mid.index.to_numpy(), size=60, replace=False)].assign(stratum='mid')]
    small = m[m['pn'] <= 10]
    pids = rng.choice(small['pid'].unique(), size=40, replace=False)
    for pid in pids:
        d = small[small['pid'] == pid]
        pick.append(d.loc[[rng.choice(d.index.to_numpy())]].assign(stratum='pointunif'))
    E = pd.concat(pick)
    Sx = []
    for lg, keys in SUSPECTS.items():
        for k in keys:
            d = m[(m['lang'] == lg) & (m['key'] == k)]
            if len(d):
                Sx.append(d.loc[rng.choice(d.index.to_numpy(), size=min(6, len(d)), replace=False)].assign(stratum='suspect'))
    X = pd.concat(Sx)
    need = set(E['doc_id']) | set(X['doc_id'])
    texts = {}
    for f in sorted((L / 'extract/texts').rglob('*.jsonl*')):
        op = gzip.open if f.suffix == '.gz' else open
        with op(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                i = line.find('"id": "')
                j = line.find('"', i + 7)
                did = line[i + 7:j]
                if did in need and did not in texts:
                    texts[did] = json.loads(line)['t']
        if len(texts) >= len(need):
            break

    def ctx(r, w=220):
        t = texts.get(r['doc_id'])
        if t is None:
            return None
        a_, b_ = int(r['char_start']), int(r['char_end'])
        return (t[max(0, a_ - w):a_] + '[[' + t[a_:b_] + ']]' + t[b_:b_ + w]).replace('\n', ' / ')

    for name, D in (('extra', E), ('suspects', X)):
        with open(O / f'{name}.jsonl', 'w', encoding='utf-8') as fh:
            for _, r in D.iterrows():
                p = P[int(r['pid'])]
                fh.write(json.dumps({'stratum': r['stratum'], 'lang': r['lang'], 'doc_id': r['doc_id'], 'year': int(r['year']) if r['year'] == r['year'] else None, 'collection': r['collection'],
                                     'surface': r['surface'], 'p_per': float(r['p_per']), 'pid': int(r['pid']), 'pname': p['name'], 'lat': p['lat'], 'lon': p['lon'], 'k': p['k'], 'r': p.get('r'),
                                     'pn': p['n'], 'gs': r['gc_source'], 'ctx': ctx(r)}, ensure_ascii=False) + '\n')
    print('done', len(E), len(X))


if __name__ == '__main__':
    main()
