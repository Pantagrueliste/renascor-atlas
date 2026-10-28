#!/usr/bin/env python3
"""v2.3 against v2.4: the numbers of the report (Mac, pandas only, a few seconds).

    .venv/bin/python -I scripts/v24_numbers.py [--v23 DIR] [--v24 DIR] [--globe_old DIR] [--globe_new DIR] [--out FILE]

Writes a JSON with: overall placed share, per edition, the gains by path (what v2.3 left unplaced and v2.4 placed, by source and tier, with the top examples), the losses (placed in
v2.3, not in v2.4) with the reason, points that moved, kinds, sources, the globe's duplicates (same display name within 5 km) and the individual errors of the check of v2.3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path('/Users/clem/Projects/Papers/RQ/derived/atlas')
S = Path('/private/tmp/claude-501/-Users-clem-Projects-Papers-Renascor/e0c9bdc2-9edc-46d5-8bde-77513f9fb822/scratchpad/atlas/globe')
KEY = ['coll_slug', 'entity_key']


def hav(a, b, c, d):
    a, b, c, d = (np.radians(np.asarray(x, dtype='float64')) for x in (a, b, c, d))
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


def load(d: Path) -> pd.DataFrame:
    want = KEY + ['name', 'n_in_period', 'coord_source', 'lat', 'lon', 'coord_kind', 'coord_conf', 'coord_radius_km', 'place_kind', 'kind_source', 'gc_source', 'gc_tier',
                  'gc_conf', 'gc_reason', 'gc_evidence', 'gc_cand_name', 'qid', 'note_type', 'global_id', 'cb_used', 'cb_geo_conf', 'cb_match', 'tier']
    have = pd.read_parquet(d / 'places_local_geo.parquet').columns
    return pd.read_parquet(d / 'places_local_geo.parquet', columns=[c for c in want if c in have])


def dup_pairs(core: dict, km: float = 5.0) -> dict:
    rows = core['rows']
    df = pd.DataFrame({'name': [str(r[0]).lower() for r in rows], 'lat': [r[1] for r in rows], 'lon': [r[2] for r in rows], 'n': [r[9] for r in rows]})
    pairs, minor = 0, 0
    ex = []
    for nm, g in df.groupby('name'):
        if len(g) < 2:
            continue
        ix = g.index.to_list()
        for i in range(len(ix)):
            for j in range(i + 1, len(ix)):
                if float(hav(df.at[ix[i], 'lat'], df.at[ix[i], 'lon'], df.at[ix[j], 'lat'], df.at[ix[j], 'lon'])) <= km:
                    pairs += 1
                    m = int(min(df.at[ix[i], 'n'], df.at[ix[j], 'n']))
                    minor += m
                    ex.append((m, nm, int(df.at[ix[i], 'n']), int(df.at[ix[j], 'n'])))
    ex.sort(reverse=True)
    return {'points': len(rows), 'same_name_pairs_within_5km': pairs, 'mentions_in_the_smaller_twin': minor, 'largest': ex[:12]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v23', type=Path, default=A / 'v2.3')
    ap.add_argument('--v24', type=Path, default=A / 'v2.4')
    ap.add_argument('--globe_old', type=Path, default=S / 'v23')
    ap.add_argument('--globe_new', type=Path, default=S / 'v24')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    g2, g3 = load(a.v23), load(a.v24)
    res: dict = {}
    for tag, g in (('v2.3', g2), ('v2.4', g3)):
        live = g['n_in_period'] > 0
        placed = g['coord_source'] != 'none'
        res[tag] = {'mentions_in_period': int(g['n_in_period'].sum()), 'mentions_placed': int(g.loc[placed, 'n_in_period'].sum()),
                    'share_placed': round(float(g.loc[placed, 'n_in_period'].sum() / max(1, g['n_in_period'].sum())), 4),
                    'local_places_in_period': int(live.sum()), 'local_places_placed': int((live & placed).sum()),
                    'placed_mentions_by_source': {k: int(v) for k, v in g[placed].groupby('coord_source')['n_in_period'].sum().items()},
                    'placed_locals_by_source': {k: int(v) for k, v in g[placed & live].groupby('coord_source').size().items()},
                    'kind_mentions': {str(k): int(v) for k, v in g[live].groupby(g['place_kind'].fillna('none'))['n_in_period'].sum().items()},
                    'coord_kind_placed_mentions': {str(k): int(v) for k, v in g[placed].groupby(g['coord_kind'].fillna('none'))['n_in_period'].sum().items()}}
        gl = json.loads((a.v23 if tag == 'v2.3' else a.v24).joinpath('globe_places.json').read_text())['places']
        res[tag]['globe_points'] = len(gl)
        res[tag]['globe_by_kind'] = pd.Series([p['k'] for p in gl]).value_counts().to_dict()
        res[tag]['globe_by_source'] = pd.Series([p['s'] for p in gl]).value_counts().to_dict()
        pg = pd.read_parquet((a.v23 if tag == 'v2.3' else a.v24) / 'places_global.parquet', columns=['n_in_period', 'lat'])
        res[tag]['global_places_in_period'] = int((pg['n_in_period'] > 0).sum())
        res[tag]['global_places_with_coords'] = int(((pg['n_in_period'] > 0) & pg['lat'].notna()).sum())
    # per edition
    def per(g):
        placed = g['coord_source'] != 'none'
        t = g.groupby('coll_slug')['n_in_period'].sum().rename('mentions')
        p = g[placed].groupby('coll_slug')['n_in_period'].sum().rename('placed')
        return pd.concat([t, p], axis=1).fillna(0)
    e2, e3 = per(g2), per(g3)
    ed = e2.join(e3, lsuffix='_23', rsuffix='_24', how='outer').fillna(0)
    ed['share_23'] = (ed['placed_23'] / ed['mentions_23'].clip(lower=1)).round(4)
    ed['share_24'] = (ed['placed_24'] / ed['mentions_24'].clip(lower=1)).round(4)
    ed['gain'] = (ed['placed_24'] - ed['placed_23']).astype(int)
    res['per_edition'] = ed.sort_values('gain', key=lambda s: -s.abs()).reset_index().astype({'mentions_23': int, 'placed_23': int, 'mentions_24': int, 'placed_24': int}).to_dict('records')
    # gains and losses at the level of local places
    # the Hearth Tax entities are keyed by file county in v2.4 ('surf:kirkby @lancashire'): their keys do not match v2.3's; they count in the totals and the per-edition table only
    ht = g2['coll_slug'].eq('hearth_tax')
    g2k, g3k = g2[~ht], g3[~g3['coll_slug'].eq('hearth_tax')]
    m = g2k.merge(g3k, on=KEY, suffixes=('_23', '_24'), how='outer', indicator=True)
    both = m[m['_merge'] == 'both'].copy()
    n23 = both['n_in_period_24'].fillna(0)
    gain = both[(both['coord_source_23'] == 'none') & (both['coord_source_24'] != 'none') & (n23 > 0)]
    loss = both[(both['coord_source_23'] != 'none') & (both['coord_source_24'] == 'none') & (both['n_in_period_23'] > 0)]
    path = gain['coord_source_24'].copy()
    tier = gain['gc_tier_24'].fillna('').astype(str)
    path = np.where(gain['coord_source_24'] == 'name_latin', 'A latin', np.where(tier.str.startswith(('polity', 'head_alt')) | (gain['coord_source_24'] == 'name_exonym'), 'B polity / qualifier / alt name',
                    np.where(tier.eq('regional_unique'), 'C regional unique', np.where(gain['coord_source_24'].isin(['name_geonames', 'name_propagation']), 'other name-based',
                    gain['coord_source_24']))))
    gain = gain.assign(path=path, mentions=gain['n_in_period_24'].astype(int))
    res['gains_by_path'] = {k: {'local_places': int(len(g)), 'mentions': int(g['mentions'].sum())} for k, g in gain.groupby('path')}
    res['gains_by_source_tier'] = {f'{s}:{t}': {'local_places': int(len(g)), 'mentions': int(g['mentions'].sum())}
                                   for (s, t), g in gain.assign(t=tier.str.replace(r'\+.*', '', regex=True)).groupby(['coord_source_24', 't'])}
    cols = ['coll_slug', 'name_24', 'mentions', 'coord_source_24', 'gc_tier_24', 'lat_24', 'lon_24', 'gc_cand_name_24', 'gc_conf_24']
    res['gains_top_examples'] = {k: g.sort_values('mentions', ascending=False).head(14)[cols].round(3).astype(object).where(lambda x: x.notna(), None).to_dict('records')
                                 for k, g in gain.groupby('path')}
    res['losses'] = {'local_places': int(len(loss)), 'mentions': int(loss['n_in_period_23'].sum()),
                     'by_reason_v24': {str(k): {'places': int(len(g)), 'mentions': int(g['n_in_period_23'].sum())} for k, g in loss.groupby(loss['gc_reason_24'].fillna('none'))},
                     'by_collection': {k: int(g['n_in_period_23'].sum()) for k, g in loss.groupby('coll_slug')},
                     'top': loss.sort_values('n_in_period_23', ascending=False).head(25)[['coll_slug', 'name_23', 'n_in_period_23', 'coord_source_23', 'lat_23', 'lon_23', 'gc_reason_24']]
                     .round(3).astype(object).where(lambda x: x.notna(), None).to_dict('records')}
    moved = both[(both['coord_source_23'] != 'none') & (both['coord_source_24'] != 'none')].copy()
    moved['km'] = hav(moved['lat_23'], moved['lon_23'], moved['lat_24'], moved['lon_24'])
    mv = moved[moved['km'] > 5]
    res['moved_more_than_5km'] = {'local_places': int(len(mv)), 'mentions': int(mv['n_in_period_24'].sum()),
                                  'by_source_24': {str(k): int(v) for k, v in mv.groupby('coord_source_24')['n_in_period_24'].sum().items()},
                                  'top': mv.sort_values('n_in_period_24', ascending=False).head(20)[['coll_slug', 'name_24', 'n_in_period_24', 'coord_source_23', 'coord_source_24', 'km']]
                                  .round(1).astype(object).where(lambda x: x.notna(), None).to_dict('records')}
    # globe duplicates
    for tag, d in (('v2.3', a.globe_old), ('v2.4', a.globe_new)):
        f = Path(d) / 'core.json'
        if f.exists():
            res[tag]['globe_duplicates'] = dup_pairs(json.loads(f.read_text()))
    # unplaced reasons
    for tag, g in (('v2.3', g2), ('v2.4', g3)):
        u = g[(g['n_in_period'] > 0) & (g['coord_source'] == 'none')]
        res[tag]['unplaced'] = {'local_places': int(len(u)), 'mentions': int(u['n_in_period'].sum()),
                                'by_reason': {str(k): int(v) for k, v in u.groupby(u['gc_reason'].fillna('not_attempted'))['n_in_period'].sum().sort_values(ascending=False).head(14).items()}}
    # individual errors of the check
    names = ['Lugduni', 'Lugdunum', 'Lugdunenses', 'Lugduno', 'Lugdunensi', 'Lugduni Bat.', 'Lugduni Batavorum', 'Traiecti', 'Traiectum', 'Augusta', 'Hadria', 'Hadriae',
             'Stolberg, Grafschaft', 'Stolberg im Harz, Grafschaft', 'Gaster, Landvogtei', 'Oberstein, Herrschaft', 'Hohenlohe, Grafschaft', 'Hohenlohe, Fürstentum',
             'Falkenberg, Herrschaft (in Oberschlesien)', 'Werdenberg, Grafschaft', 'Coronia', 'AEgea', 'Aegaeae', 'Becam', 'Nordovico', 'Palatinus', 'Fagum', 'Tagum', 'Lucena (Córdoba)',
             'Lucena', 'Kirkby', 'Heath', 'Italia, Milán', 'Italia , Milán', 'Cross Keys Inn (Holborn)', 'Cross Keys Inn (St. John’s Street)', 'Cross Keys Inn (Gracechurch Street)',
             'Thorney', 'St. Pancras (Soper Lane)', 'Parish of St. Pancras (Soper Lane)', 'New-street', 'St. Helens', 'Bow', 'Samothraciam', 'Jettenbach, Schloss', 'Europe', 'Europa', 'Asia',
             'Africa', 'Amerika', 'Vienne', 'Indias', 'The Hythe']
    chk = []
    for nm in names:
        for tag, g in (('v2.3', g2), ('v2.4', g3)):
            r = g[(g['name'] == nm) & (g['n_in_period'] > 0)].sort_values('n_in_period', ascending=False).head(3)
            for _, x in r.iterrows():
                chk.append({'version': tag, 'coll': x['coll_slug'], 'name': nm, 'n': int(x['n_in_period']), 'source': x['coord_source'], 'lat': None if pd.isna(x['lat']) else round(float(x['lat']), 3),
                            'lon': None if pd.isna(x['lon']) else round(float(x['lon']), 3), 'kind': x.get('place_kind'), 'tier': x.get('gc_tier')})
    res['named_cases'] = chk
    t = json.dumps(res, indent=1, ensure_ascii=False, default=str)
    if a.out:
        a.out.write_text(t)
    print(t[:6000])


if __name__ == '__main__':
    main()
