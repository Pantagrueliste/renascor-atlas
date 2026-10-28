"""NER layer v1, geocoding stage 5 (Mac, nice): summary tables of the layer into RQ/derived/atlas/ner/v2/layer/.

    nice -n 19 python scripts/ner2_layer_geo_tables.py --work DIR --model-layer model_layer.json --out OUTDIR

DIR holds the files copied from ``geo/points`` and ``geo/gc`` of Hypatia (forms_placed.parquet, fates.tsv, fates_by_lang.tsv, placed_by_collection.tsv, gc_run.json, summary.json, match_summary.json).
Writes geocode_fates.tsv, geocode_fates_by_lang.tsv, geocode_placed_by_lang.tsv, geocode_forms.tsv.gz (every form and what became of it), geocode_points.tsv.gz,
geocode_match_summary.json, geocode_placed_by_collection.tsv, geocode_run.json, geocode_summary.json, geocode_points_stats.json.
"""
import argparse
import collections
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--model-layer', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    W, OUT = a.work, a.out
    F = pd.read_parquet(W / 'forms_placed.parquet')
    pts = json.loads(a.model_layer.read_text('utf-8'))['places']
    fates = pd.read_csv(W / 'fates.tsv', sep='\t')
    fates['share_mentions'] = (fates.mentions / fates.mentions.sum()).round(4)
    fates.to_csv(OUT / 'geocode_fates.tsv', sep='\t', index=False)
    shutil.copy(W / 'fates_by_lang.tsv', OUT / 'geocode_fates_by_lang.tsv')
    rows = []
    for lg, d in F.groupby('lang'):
        pl = d[d.fate == 'placed']
        rows.append(dict(lang=lg, forms=len(d), mentions=int(d.n.sum()), forms_placed=len(pl), mentions_placed=int(pl.n.sum()), share_placed=round(pl.n.sum() / d.n.sum(), 4)))
    t = pd.DataFrame(rows)
    tot = dict(lang='total', forms=t.forms.sum(), mentions=t.mentions.sum(), forms_placed=t.forms_placed.sum(), mentions_placed=t.mentions_placed.sum())
    tot['share_placed'] = round(tot['mentions_placed'] / tot['mentions'], 4)
    t = pd.concat([t, pd.DataFrame([tot])])
    t.to_csv(OUT / 'geocode_placed_by_lang.tsv', sep='\t', index=False)
    cols = ['lang', 'key', 'form', 'n', 'n_docs', 'fate', 'pid', 'kind', 'gc_lat', 'gc_lon', 'gc_source', 'gc_conf', 'gc_radius_km', 'gc_cand_name', 'gc_reason']
    F[cols].sort_values('n', ascending=False).to_csv(OUT / 'geocode_forms.tsv.gz', sep='\t', index=False, compression='gzip')
    rows = []
    for p in pts:
        e = p.get('ed') or {}
        rows.append(dict(id=p['id'], name=p['name'], lat=p['lat'], lon=p['lon'], kind=p['k'], radius_km=p.get('r'), mentions=p['n'], docs=p['nd'], method=p['gs'], conf=p['conf'],
                         forms=' | '.join(f'{f[0]} ({f[1]})' for f in p['forms'][:5]), editor_id=e.get('id'), editor_name=e.get('name'), editor_km=e.get('km'), editor_how=e.get('how')))
    pd.DataFrame(rows).to_csv(OUT / 'geocode_points.tsv.gz', sep='\t', index=False, compression='gzip')
    st = {'points': len(pts), 'mentions': sum(p['n'] for p in pts)}
    for key, fn in (('by_kind', lambda p: p['k']), ('by_method', lambda p: p['gs'])):
        c, m = collections.Counter(), collections.Counter()
        for p in pts:
            c[fn(p)] += 1
            m[fn(p)] += p['n']
        st[key] = {k: [c[k], m[k]] for k in c}
    st['mentions_by_conf'] = dict(collections.Counter())
    cf = collections.Counter()
    for p in pts:
        cf[p['conf']] += p['n']
    st['mentions_by_conf'] = dict(cf)
    st['quarters_1450_by_25'] = [sum(p['q'][i] for p in pts) for i in range(10)]
    st['points_n_ge_10'] = sum(p['n'] >= 10 for p in pts)
    st['points_n_ge_100'] = sum(p['n'] >= 100 for p in pts)
    st['points_n_1'] = sum(p['n'] == 1 for p in pts)
    st['points_one_document'] = sum(p['nd'] == 1 for p in pts)
    (OUT / 'geocode_points_stats.json').write_text(json.dumps(st, indent=1), 'utf-8')
    for src, dst in (('match_summary.json', 'geocode_match_summary.json'), ('placed_by_collection.tsv', 'geocode_placed_by_collection.tsv'), ('gc_run.json', 'geocode_run.json'), ('summary.json', 'geocode_summary.json')):
        shutil.copy(W / src, OUT / dst)
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
