#!/usr/bin/env python3
"""v2.1 against v2.2: the numbers of the report (Mac, pandas only).  .venv/bin/python -I scripts/v22_numbers.py [--v21 DIR] [--v22 DIR] [--out FILE]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path('/Users/clem/Projects/Papers/RQ/derived/atlas')


def hav(a, b, c, d):
    a, b, c, d = (np.radians(np.asarray(x, dtype='float64')) for x in (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def one(d: Path) -> dict:
    g = pd.read_parquet(d / 'places_local_geo.parquet')
    live = g['n_in_period'] > 0
    placed = g['coord_source'] != 'none'
    out = {'mentions_in_period': int(g['n_in_period'].sum()), 'mentions_placed': int(g.loc[placed, 'n_in_period'].sum()),
           'local_places_placed': int((live & placed).sum()), 'local_places_in_period': int(live.sum())}
    out['share_placed'] = round(out['mentions_placed'] / max(out['mentions_in_period'], 1), 4)
    out['placed_by_source'] = {k: int(v) for k, v in g[placed].groupby('coord_source')['n_in_period'].sum().items()}
    gl = json.loads((d / 'globe_places.json').read_text())
    pts = gl['places']
    out['globe_points'] = len(pts)
    out['globe_by_kind'] = pd.Series([p['k'] for p in pts]).value_counts().to_dict()
    out['globe_by_source'] = pd.Series([p['s'] for p in pts]).value_counts().to_dict()
    pg = pd.read_parquet(d / 'places_global.parquet', columns=['n_in_period', 'lat'])
    out['global_places_in_period'] = int((pg['n_in_period'] > 0).sum())
    out['global_places_with_coords'] = int(((pg['n_in_period'] > 0) & pg['lat'].notna()).sum())
    gp = pd.read_parquet(d / 'gate_pairs.parquet', columns=['kind', 'scheme', 'status', 'reason'])
    out['gate_pairs'] = gp.groupby(['kind', 'status']).size().rename(lambda x: x).reset_index().assign(k=lambda t: t['kind'] + ':' + t['status']).set_index('k')[0].to_dict()
    out['gate_reasons_places'] = {f'{s}:{r}': int(n) for (s, r), n in gp[(gp['kind'] == 'place') & (gp['status'] != 'accepted')].groupby(['status', 'reason']).size().items()}
    pe = pd.read_parquet(d / 'persons_global.parquet', columns=['n_in_period', 'qid'])
    out['persons_global_in_period'] = int((pe['n_in_period'] > 0).sum())
    out['persons_global_with_qid'] = int(pe['qid'].notna().sum())
    out['place_kind_local'] = g[live].groupby(g['place_kind'].fillna('none'))['n_in_period'].sum().to_dict()
    # towns typed as areas (v2.1 bug): local places whose GeoNames feature is a settlement of 1000+ and whose kind is area
    town = g['gn_fcode'].astype(str).str.startswith('PPL') & (g['gn_pop'].fillna(0) >= 1000)
    out['towns_typed_area'] = {'places': int((live & town & (g['place_kind'] == 'area')).sum()), 'mentions': int(g.loc[live & town & (g['place_kind'] == 'area'), 'n_in_period'].sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v21', type=Path, default=A / 'v2.1')
    ap.add_argument('--v22', type=Path, default=A / 'v2.2')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    res = {'v2.1': one(a.v21), 'v2.2': one(a.v22)}
    for v, d in (('v2.1', a.v21), ('v2.2', a.v22)):
        r = json.loads((d / 'reconcile_report.json').read_text())
        res[v]['excluded_mentions'] = r.get('excluded_mentions', {}).get('by_first_reason', {})
        res[v]['excluded_in_period'] = r.get('excluded_mentions', {}).get('by_first_reason_in_period', {})
    t = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    print(t)
    if a.out:
        a.out.write_text(t)


if __name__ == '__main__':
    main()
