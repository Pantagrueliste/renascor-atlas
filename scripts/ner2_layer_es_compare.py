"""Spanish rerun: old (xlmr main, layer_v1) against new (mDeBERTa main, es_mdeberta) Spanish spans, recommended spans, placed mentions and points (Hypatia, CPU).

    python scripts/ner2_layer_es_compare.py --layer LAYER_V1 --es ES_DIR --out ES_DIR/compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def rec(s, min_conf, min_agree):
    return s[(s['conf'] >= min_conf) & (s['n_agree'].fillna(0) >= min_agree)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=Path, required=True)
    ap.add_argument('--es', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    cols = ['doc_id', 'char_start', 'char_end', 'surface', 'conf', 'n_agree', 'lang']
    old = pd.read_parquet(a.layer / 'spans.parquet', columns=cols, filters=[('lang', '=', 'es')])
    new = pd.read_parquet(a.es / 'spans.parquet', columns=cols)
    R = {'old_docs': int(old['doc_id'].nunique()), 'new_docs': int(new['doc_id'].nunique())}
    R['spans'] = {'old': len(old), 'new': len(new)}
    R['conf_ge_098'] = {'old': int((old['conf'] >= 0.98).sum()), 'new': int((new['conf'] >= 0.98).sum())}
    R['recommended_rule'] = {'old (xlmr main, conf>=0.98, agree>=2)': len(rec(old, 0.98, 2)), 'new strict rule (mdeberta main, conf>=0.98, agree>=2)': len(rec(new, 0.98, 2)),
                             'new chosen rule (mdeberta main, conf>=0.5, agree>=2)': len(rec(new, 0.5, 2)), 'new agree>=3': len(rec(new, 0.5, 3)), 'new agree>=4': len(rec(new, 0.5, 4))}
    ro, rn = rec(old, 0.98, 2), rec(new, 0.5, 2)
    ko = set(zip(ro['doc_id'], ro['char_start'], ro['char_end']))
    kn = set(zip(rn['doc_id'], rn['char_start'], rn['char_end']))
    R['recommended_overlap_exact'] = {'both': len(ko & kn), 'old_only': len(ko - kn), 'new_only': len(kn - ko)}
    # forms
    Fo = pd.read_parquet(a.layer / 'geo/points/forms_placed.parquet')
    Fo = Fo[Fo['lang'] == 'es']
    Fn = pd.read_parquet(a.es / 'geo/points/forms_placed.parquet')
    def fs(F):
        pl = F[F['fate'] == 'placed']
        return {'forms': len(F), 'mentions': int(F['n'].sum()), 'forms_placed': len(pl), 'mentions_placed': int(pl['n'].sum()), 'share_placed': float(pl['n'].sum() / F['n'].sum())}
    R['forms'] = {'old': fs(Fo), 'new': fs(Fn)}
    po, pn = set(Fo[Fo['fate'] == 'placed']['key']), set(Fn[Fn['fate'] == 'placed']['key'])
    R['placed_forms_overlap'] = {'both': len(po & pn), 'old_only': len(po - pn), 'new_only': len(pn - po)}
    both = Fo[Fo['fate'] == 'placed'].merge(Fn[Fn['fate'] == 'placed'], on='key', suffixes=('_o', '_n'))
    import numpy as np
    d = np.hypot((both['gc_lat_o'] - both['gc_lat_n']), (both['gc_lon_o'] - both['gc_lon_n']))
    R['placed_forms_same_coords_in_both'] = int((d < 0.01).sum())
    R['fates_new'] = Fn.groupby('fate')['n'].agg(['size', 'sum']).rename(columns={'size': 'forms', 'sum': 'mentions'}).reset_index().to_dict('records')
    R['fates_old'] = Fo.groupby('fate')['n'].agg(['size', 'sum']).rename(columns={'size': 'forms', 'sum': 'mentions'}).reset_index().to_dict('records')
    # points: old points that carry Spanish mentions (they are shared with other languages), new points (Spanish only)
    Po = json.loads((a.layer / 'geo/points/points.json').read_text('utf-8'))
    Pn = json.loads((a.es / 'geo/points/points.json').read_text('utf-8'))
    pes = [p for p in Po if p['langs'].get('es')]
    R['points'] = {'old_with_spanish_mentions': len(pes), 'old_spanish_mentions_in_points': sum(p['langs']['es'] for p in pes),
                   'old_points_mostly_spanish': sum(1 for p in pes if p['langs']['es'] >= 0.5 * p['n']), 'old_points_spanish_only': sum(1 for p in pes if set(p['langs']) == {'es'}),
                   'new_points': len(Pn), 'new_mentions': sum(p['n'] for p in Pn), 'new_points_n_ge_10': sum(1 for p in Pn if p['n'] >= 10),
                   'new_points_n_1': sum(1 for p in Pn if p['n'] == 1)}
    on = {(round(p['lat'], 1), round(p['lon'], 1)) for p in pes if set(p['langs']) == {'es'}}
    R['points']['old_spanish_only_points_n_le_10'] = sum(1 for p in pes if set(p['langs']) == {'es'} and p['n'] <= 10)
    R['points']['new_points_n_le_10'] = sum(1 for p in Pn if p['n'] <= 10)
    # by collection: recommended
    R['recommended_by_collection'] = {'old': ro.merge(pd.read_parquet(a.layer / 'docs.parquet', columns=['doc_id', 'collection']), on='doc_id').groupby('collection').size().to_dict(),
                                      'new': rn.merge(pd.read_parquet(a.layer / 'docs.parquet', columns=['doc_id', 'collection']), on='doc_id').groupby('collection').size().to_dict()}
    a.out.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=int), 'utf-8')
    print(json.dumps(R, indent=1, ensure_ascii=False, default=int))


if __name__ == '__main__':
    main()
