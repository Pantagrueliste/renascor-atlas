"""Development aid (v2.3): where does the name-based geocoder send these names? Runs on Hypatia through Slurm (``slurm/v23.sbatch trace NAME...``), never on the login node.

    python -I scripts/geocode_trace.py --geo places_local_geo.parquet --local places_local.parquet --mentions mentions.parquet --index gn_index.parquet \
        [--orbis data/orbis_latinus.tsv --wd-entities wd_entities.parquet] [--limit 4] "Leidam" "ckcc_epistolarium:Hagam" "Löbau (Lubawa)"

A query is a name, or ``collection:name``. For every matching place (the most mentioned first) it prints the paths: the polity table, the exonym table, the generic
lookup (propagation and GeoNames), the regional rule, the Latin path, and the final answer with its evidence. Nothing is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ratlas import reconcile_geocode as GC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--geo', type=Path, required=True)
    ap.add_argument('--local', type=Path, required=True)
    ap.add_argument('--mentions', type=Path, required=True)
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--orbis', type=Path, default=None)
    ap.add_argument('--wd-entities', type=Path, default=None, dest='wd_entities')
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('queries', nargs='+')
    a = ap.parse_args(argv)
    P = GC.load_places(a.geo, a.local)
    pairs, hint, doc_regions = GC.load_pairs(a.mentions, P)
    gn = GC.GN(pd.read_parquet(a.index))
    res = GC.load_resources(a)
    res['mention_aux'] = GC.load_mention_aux(a.mentions, P)          # v2.4: the passages of the Latin homonyms, the surface check
    G = GC.Geocoder(P, gn, pairs, hint, doc_regions, res)
    lat0, lon0 = P['lat'].to_numpy('float64').copy(), P['lon'].to_numpy('float64').copy()
    G.set_anchors(lat0, lon0)
    G.set_context(lat0, lon0, np.log1p(G.weight))
    for q in a.queries:
        coll, _, name = q.partition(':') if ':' in q and q.split(':', 1)[0] in set(P['coll_slug']) else (None, '', q)
        m = (P['name'].str.casefold() == name.casefold())
        if coll:
            m &= (P['coll_slug'] == coll)
        rows = P[m].sort_values('n_in_period', ascending=False).head(a.limit)
        print(f'=== {q!r}: {int(m.sum())} places')
        for i, r in rows.iterrows():
            print(f"--- [{i}] {r['coll_slug']} {r['entity_key']} {r['name']!r} n_in_period={r['n_in_period']} note={r['note_type']} src0={r['src0']} "
                  f"latin={G.latin_level[i]}")
            if np.isfinite(r['lat']):
                print(f"    placed already: {r['lat']:.4f} {r['lon']:.4f} ({r['coord_source']})")
                continue
            for what, fn in (('polity', G.polity), ('exonym', G.exonym), ('polity_gn', G.polity_gn), ('head_alt', G.head_alt), ('latin', G.latin)):
                try:
                    x = fn(i)
                except Exception as e:                       # noqa: BLE001
                    x = {'conf': 'ERROR', 'evidence': repr(e), 'lat': float('nan'), 'lon': float('nan')}
                if x:
                    print(f"    {what:10s} {x['conf']} {x['lat']:.3f} {x['lon']:.3f} r={x.get('radius')} {x.get('tier')}: {str(x['evidence'])[:300]}")
            if G.latin_on and G.latin_level[i] in ('latin', 'mixed'):
                from ratlas import latin_names as LN
                for c in LN.lemmatize(str(r['name']), min_score=0.4):
                    qs = sorted(G.wdlatin.get(c.lemma)) if G.wdlatin is not None else []
                    es = G.orbis.get(c.lemma) if G.orbis is not None else []
                    print(f"    lemma {c.lemma!r} {c.score:.2f} {c.rule}: wd {qs} placed={[q for q in qs if q in G.wd_points]} orbis={[(e.modern, sorted(e.ccs or [])) for e in es][:2]} "
                          f"atlas_rows={len(G.RL.get(LN.fold_key(c.lemma), []))}")
            res = G.resolve(i)
            for k in ('prop', 'gn'):
                x = res.get(k)
                if x:
                    print(f"    {k:10s} {x.get('conf')} {x.get('lat', float('nan')):.3f} {x.get('lon', float('nan')):.3f} {x.get('tier')}: {str(x.get('evidence') or x.get('reason'))[:260]}")
            f = res['final']
            if f:
                print(f"    FINAL      {f['conf']} {f['method']} {f['lat']:.4f} {f['lon']:.4f} r={f.get('radius')} via={res['via']} cand={f.get('cand_name')}: {str(f['evidence'])[:260]}")
            else:
                print(f"    FINAL      none; why={G.why.get(i)} via={res['via']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
