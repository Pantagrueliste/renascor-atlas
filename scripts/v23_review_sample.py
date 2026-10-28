#!/usr/bin/env python3
"""Read-through of the new name-based placements of v2.3 (Mac, pandas only): a seeded sample per path (Latin, polity / qualifier / alternative name, regional homonym), each with
the name, the answer, the evidence and the distance to the nearest same-name feature of the sample's own edition. The sample is read by a person; the verdicts go to the report.

    .venv/bin/python -I scripts/v23_review_sample.py [--v23 DIR] [--n 25] [--seed 7] [--out FILE]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

A = Path('/Users/clem/Projects/Papers/RQ/derived/atlas')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v23', type=Path, default=A / 'v2.3')
    ap.add_argument('--v22', type=Path, default=A / 'v2.2')
    ap.add_argument('--n', type=int, default=25)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    cols = ['coll_slug', 'entity_key', 'name', 'n_in_period', 'coord_source', 'coord_conf', 'lat', 'lon', 'gc_tier', 'gc_cand_name', 'gc_evidence', 'gc_conf', 'coord_kind', 'coord_radius_km']
    g3 = pd.read_parquet(a.v23 / 'places_local_geo.parquet', columns=cols)
    g2 = pd.read_parquet(a.v22 / 'places_local_geo.parquet', columns=['coll_slug', 'entity_key', 'coord_source'])
    new = g3.merge(g2, on=['coll_slug', 'entity_key'], how='left', suffixes=('', '_22'))
    new = new[(new['coord_source_22'] == 'none') & (new['coord_source'] != 'none') & (new['n_in_period'] > 0)]
    tier = new['gc_tier'].fillna('').astype(str)
    paths = {'A latin': new[new['coord_source'] == 'name_latin'],
             'B polity / qualifier / alt': new[tier.str.startswith(('polity', 'head_alt'))],
             'C regional unique': new[tier.eq('regional_unique')]}
    lines = []
    for k, d in paths.items():
        lines.append(f'## {k}: {len(d)} local places, {int(d["n_in_period"].sum())} mentions')
        s = d.sample(min(a.n, len(d)), random_state=a.seed).sort_values('n_in_period', ascending=False)
        for r in s.itertuples():
            lines.append(f"- [{r.coll_slug}] {r.name!r} x{r.n_in_period} -> {r.gc_cand_name!r} ({r.lat:.3f}, {r.lon:.3f}) {r.coord_conf} {r.coord_kind} {r.gc_tier}: {str(r.gc_evidence)[:220]}")
        lines.append('')
    t = '\n'.join(lines)
    print(t)
    if a.out:
        a.out.write_text(t)


if __name__ == '__main__':
    main()
