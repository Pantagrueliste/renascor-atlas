"""v1 / v1.1 / v2 / v2.1 / v2.2 / v2.3 side by side: mentions placed, distinct places, globe points, persons with an accepted authority.

    .venv/bin/python -I scripts/compare_versions.py [--atlas DIR] [--v2 DIR] [--v21 DIR] [--out FILE]

Reads the parquet / json tables of ``RQ/derived/atlas/{v1,v1.1,v2,v2.1}`` and writes ``v2.1/version_comparison.json`` (``--v21`` absent: the v2
file as before, three versions). Same definitions for the versions (in-period mentions of the table; v2 and v2.1 counts are the atlas counts,
``_all`` is the canonical total). The placed mentions are also split by coordinate source. Pandas only, a few seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ATLAS = Path('/Users/clem/Projects/Papers/RQ/derived/atlas')


def places(d: Path, v2: bool) -> dict:
    g = pd.read_parquet(d / 'places_local_geo.parquet', columns=['n_in_period', 'coord_source', 'qid', 'scope'] + (['n_in_period_all'] if v2 else []))
    placed = g['coord_source'] != 'none'
    out = {'local_places_in_period': int((g['n_in_period'] > 0).sum()),
           'mentions_in_period': int(g['n_in_period'].sum()),
           'mentions_in_period_placed': int(g.loc[placed, 'n_in_period'].sum()),
           'local_places_in_period_placed': int(((g['n_in_period'] > 0) & placed).sum()),
           'mentions_in_period_with_qid': int(g.loc[g['qid'].notna(), 'n_in_period'].sum())}
    out['share_placed_mentions'] = round(out['mentions_in_period_placed'] / max(1, out['mentions_in_period']), 4)
    out['mentions_in_period_placed_by_coord_source'] = {str(k): int(v) for k, v in g[placed].groupby('coord_source')['n_in_period'].sum().items()}
    out['local_places_in_period_placed_by_coord_source'] = {str(k): int(v) for k, v in g[placed & (g['n_in_period'] > 0)].groupby('coord_source').size().items()}
    if v2:
        out['mentions_in_period_all_canonical'] = int(g['n_in_period_all'].sum())
        out['mentions_in_period_all_canonical_placed'] = int(g.loc[placed, 'n_in_period_all'].sum())
    pg = pd.read_parquet(d / 'places_global.parquet', columns=['n_in_period', 'lat', 'qid'])
    out['distinct_places_global_in_period'] = int((pg['n_in_period'] > 0).sum())
    out['distinct_places_global_in_period_with_coords'] = int(((pg['n_in_period'] > 0) & pg['lat'].notna()).sum())
    gl = json.loads((d / 'globe_places.json').read_text())
    out['globe_points'] = len(gl['places'])
    out['globe_mentions'] = int(sum(p['n'] for p in gl['places']))
    return out


def persons(d: Path, v2: bool) -> dict:
    p = pd.read_parquet(d / 'persons_local_global.parquet', columns=['n_in_period', 'qid', 'scope'])
    live = p['n_in_period'] > 0
    out = {'local_persons_in_period': int(live.sum()), 'local_persons_in_period_with_qid': int((live & p['qid'].notna()).sum()),
           'mentions_in_period': int(p['n_in_period'].sum()), 'mentions_in_period_with_qid': int(p.loc[p['qid'].notna(), 'n_in_period'].sum())}
    pg = pd.read_parquet(d / 'persons_global.parquet', columns=['n_in_period', 'qid'] + (['wd_human'] if 'wd_human' in pd.read_parquet(d / 'persons_global.parquet').columns else []))
    out['persons_global_in_period'] = int((pg['n_in_period'] > 0).sum())
    out['persons_global_with_qid'] = int(pg['qid'].notna().sum())
    if 'wd_human' in pg:
        out['persons_global_with_non_human_qid'] = int((pg['wd_human'].astype(str) == 'False').sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', type=Path, default=ATLAS)
    ap.add_argument('--v2', type=Path, default=None)
    ap.add_argument('--v21', type=Path, default=None, help='the v2.1 folder (default: <atlas>/v2.1 when it exists)')
    ap.add_argument('--v22', type=Path, default=None, help='the v2.2 folder (default: <atlas>/v2.2 when it exists)')
    ap.add_argument('--v23', type=Path, default=None, help='the v2.3 folder (default: <atlas>/v2.3 when it exists)')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    v2 = a.v2 or (a.atlas / 'v2')
    v21 = a.v21 or ((a.atlas / 'v2.1') if (a.atlas / 'v2.1' / 'places_local_geo.parquet').exists() else None)
    v22 = a.v22 or ((a.atlas / 'v2.2') if (a.atlas / 'v2.2' / 'places_local_geo.parquet').exists() else None)
    v23 = a.v23 or ((a.atlas / 'v2.3') if (a.atlas / 'v2.3' / 'places_local_geo.parquet').exists() else None)
    versions = [('v1', a.atlas / 'v1', False), ('v1.1', a.atlas / 'v1.1', False), ('v2', v2, True)] + ([('v2.1', v21, True)] if v21 else []) \
        + ([('v2.2', v22, True)] if v22 else []) + ([('v2.3', v23, True)] if v23 else [])
    res = {}
    for name, d, is2 in versions:
        res[name] = {'places': places(d, is2), 'persons': persons(d, is2)}
    for name, d, is2 in versions:
        if is2 and (d / 'reconcile_report.json').exists():
            rep = json.loads((d / 'reconcile_report.json').read_text())
            res[name]['gates'] = rep.get('gates', {})
            for k in ('excluded_mentions', 'casebooks_directory', 'manual_overrides', 'exonym_qid_attached', 'qualifier_kind', 'kind_overrides'):
                if k in rep:
                    res[name][k] = rep[k]
    if v21 and (v21 / 'calibration_v2_1.json').exists():
        res['v2.1']['calibration'] = json.loads((v21 / 'calibration_v2_1.json').read_text())
    if v22 and (v22 / 'calibration_v2_1.json').exists():
        res['v2.2']['calibration'] = json.loads((v22 / 'calibration_v2_1.json').read_text())
    out = a.out or ((v23 or v22 or v21 or v2) / 'version_comparison.json')
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    names = [v[0] for v in versions]
    for k in ('places', 'persons'):
        print(k)
        print(pd.DataFrame({v: {kk: vv for kk, vv in res[v][k].items() if not isinstance(vv, dict)} for v in names}).to_string())


if __name__ == '__main__':
    main()
