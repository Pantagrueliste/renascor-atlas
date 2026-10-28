"""Spanish rerun: draw random placed mentions of the new layer and cut their passages (Hypatia, CPU; the text stays on Hypatia unless you copy the output).

    python scripts/ner2_layer_es_sample.py --es ES_DIR --n 40 --seed 42 --out ES_DIR/review/sample.jsonl [--ctx 220]

A mention = a recommended span (conf >= min-conf, n_agree >= min-agree) whose cleaned form was placed (fate 'placed' in geo/points/forms_placed.parquet).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ner2_layer_geo_prep import clean_surface


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval (lo, hi) for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--es', type=Path, required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-conf', type=float, default=0.5)
    ap.add_argument('--min-agree', type=int, default=2)
    ap.add_argument('--max-form-n', type=int, default=0, help='only forms with at most this many mentions (the tail of rare forms); 0 = all')
    ap.add_argument('--geo-sub', default='geo', help='geo directory under --es (geo or geo_strict)')
    ap.add_argument('--ctx', type=int, default=220)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    s = pd.read_parquet(a.es / 'spans.parquet')
    s = s[(s['conf'] >= a.min_conf) & (s['n_agree'].fillna(0) >= a.min_agree)].copy()
    s['form'] = s['surface'].map({u: clean_surface(u) for u in s['surface'].unique()})
    s['key'] = s['form'].str.casefold()
    F = pd.read_parquet(a.es / a.geo_sub / 'points/forms_placed.parquet')
    F = F[F['fate'] == 'placed'][['lang', 'key', 'n', 'pid', 'gc_cand_name', 'gc_lat', 'gc_lon', 'gc_source', 'gc_conf']]
    m = s.merge(F, on=['lang', 'key'])
    if a.max_form_n:
        m = m[m['n'] <= a.max_form_n].reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    pick = m.iloc[np.sort(rng.choice(len(m), size=min(a.n, len(m)), replace=False))].copy()
    need = set(pick['doc_id'])
    text = {}
    for f in sorted((a.es / 'extract/texts').glob('*/*.jsonl.gz')):
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                if not need:
                    break
                d = json.loads(line)
                if d['id'] in need and d['id'] not in text:
                    text[d['id']] = d['t']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as o:
        for i, (_, r) in enumerate(pick.iterrows()):
            t = text.get(r['doc_id'], '')
            cs, ce = int(r['char_start']), int(r['char_end'])
            ctx = t[max(0, cs - a.ctx):cs] + '[[' + t[cs:ce] + ']]' + t[ce:ce + a.ctx]
            o.write(json.dumps({'i': i, 'doc_id': r['doc_id'], 'collection': r['collection'], 'year': int(r['year']) if pd.notna(r['year']) else None, 'surface': r['surface'],
                                'conf': float(r['conf']), 'n_agree': int(r['n_agree']), 'pname': r['gc_cand_name'], 'lat': float(r['gc_lat']), 'lon': float(r['gc_lon']),
                                'gs': r['gc_source'], 'gconf': r['gc_conf'], 'pid': int(r['pid']), 'form_n': int(r['n']), 'ctx': ctx}, ensure_ascii=False) + '\n')
    print(len(m), 'placed mentions;', len(pick), 'drawn ->', a.out)


if __name__ == '__main__':
    main()
