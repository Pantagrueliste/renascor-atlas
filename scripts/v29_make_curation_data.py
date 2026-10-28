"""Version 29 curation data (Mac): turns the verified work of 2026-10-28 (RQ/derived/atlas/v2.9/*) into the durable files of ``data/curation/``.

    python scripts/v29_make_curation_data.py [--rq RQ_DERIVED_ATLAS] [--layerwork DIR_WITH_V1_ES_TABLES]

Writes
  data/curation/title_excluded_editor.parquet  editor mentions inside a title (row key of v2.4/mentions.parquet)
  data/curation/title_excluded_model.tsv.gz    model mentions inside a title ((src, doc_id, lang, key) of the layer's doc_forms)
  data/curation/place_merges.json              merged points, label fixes, QID fixes; model points are named by ``model_keys`` (their ``prev`` lists)
  data/curation/dicotopo_names.tsv             DicoTopo display names, written forms, article id and INSEE code of 832 points
  data/curation/place_keep_apart.tsv           pairs read and kept apart (deny list for any later automatic same-name pass)
The ids ``e:N`` are rows of D.rows of the page build after the homonym audit (page_kit/v28/build_b); ``ml2:N`` are points of the layer
page_kit/v29/model_layer_v2.json. The ml2 ids move whenever the layer is rebuilt, so every ml2 id used by these files is also given as its
``prev`` list (the units of the layer it was made of): ``ratlas.placemerge.match_model_points`` finds the point again in a newer layer.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CUR = ROOT / 'data' / 'curation'
sys.path.insert(0, str(ROOT / 'src'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rq', default='/Users/clem/Projects/Papers/RQ/derived/atlas')
    ap.add_argument('--layerwork', required=True, help='folder with doc_forms.parquet, es/doc_forms.parquet (tables of the v1 and Spanish runs)')
    a = ap.parse_args()
    R = Path(a.rq)
    V = R / 'v2.9'

    # ---------------------------------------------------------------------------------------------------- titles
    e = pd.read_parquet(V / 'titles' / 'excluded_editor_mentions.parquet')
    keep = ['row', 'coll_slug', 'doc_id', 'surface', 'lang', 'category', 'title_word', 'rule']
    e[keep].sort_values('row').to_parquet(CUR / 'title_excluded_editor.parquet', index=False)
    m = pd.read_parquet(V / 'titles' / 'excluded_model.parquet')
    m = m[['src', 'doc_id', 'lang', 'key', 'n_total', 'n_royal', 'n_noble', 'n_ecclesiastical']].copy()
    miss = pd.read_csv(V / 'titles' / 'corrections' / 'model_spans_missed_by_surname_rule.tsv', sep='\t', keep_default_na=False)
    W = Path(a.layerwork)
    tot_src = {'ml': W / 'doc_forms.parquet', 'mles': W / 'es' / 'doc_forms.parquet', 'ml3': R / 'ner' / 'v2' / 'layer' / 'v3' / 'geo_all' / 'doc_forms.parquet'}
    totals = {s: pd.read_parquet(p).set_index(['doc_id', 'lang', 'key'])['n'].to_dict() for s, p in tot_src.items()}
    idx = {(r.src, r.doc_id, r.lang, r.key): i for i, r in enumerate(m.itertuples())}
    colmap = {'royal': 'n_royal', 'noble': 'n_noble', 'ecclesiastical': 'n_ecclesiastical'}
    added = 0
    for r in miss.itertuples():
        k = (r.src, r.doc_id, r.lang, r.key)
        if k not in idx:
            n_total = totals[r.src][(r.doc_id, r.lang, r.key)]
            m.loc[len(m)] = [r.src, r.doc_id, r.lang, r.key, n_total, 0, 0, 0]
            idx[k] = len(m) - 1
        m.loc[idx[k], colmap[r.category]] += 1
        added += 1
    assert (m['n_royal'] + m['n_noble'] + m['n_ecclesiastical'] <= m['n_total']).all()
    m.sort_values(['src', 'doc_id', 'lang', 'key']).to_csv(CUR / 'title_excluded_model.tsv.gz', sep='\t', index=False)
    print('titles: editor', len(e), 'model lines', len(m), 'spans added from the surname-rule correction', added)

    # ---------------------------------------------------------------------------------------------------- model point keys
    layer = json.loads((R / 'page_kit' / 'v29' / 'model_layer_v2.json').read_text('utf-8'))['places']
    prev = {p['id']: sorted(p['prev']) for p in layer}

    # ---------------------------------------------------------------------------------------------------- merges, labels
    mg = json.loads((V / 'duplicates' / 'merges.json').read_text('utf-8'))
    lf = json.loads((V / 'duplicates' / 'label_fixes.json').read_text('utf-8'))
    merges = []
    for x in mg:
        merges.append({'keep': x['keep'], 'absorb': x['absorb'], 'reading': x['reading'], 'reason': x['reason'], 'source': 'duplicates',
                       'casebooks': bool(x.get('casebooks')), 'new_label': x.get('new_label'),
                       'info': {k: {'name': v['name'], 'lat': v['lat'], 'lon': v['lon']} for k, v in
                                [(x['keep_info']['id'], x['keep_info'])] + [(i['id'], i) for i in x['absorb_info']]}})
    # DicoTopo: points of the same article that the build does not merge yet
    mc = pd.read_csv(V / 'dicotopo' / 'merge_candidates_corrected.tsv', sep='\t', keep_default_na=False)
    mc = mc[(mc['already_merged'] == 'no') & (mc['valid'] == 'yes')]
    dm = pd.read_csv(V / 'dicotopo' / 'matches_corrected.tsv', sep='\t', keep_default_na=False)
    dm = dm[dm['status'] == 'apply']
    info_dt = {r.point_id: {'name': r.current_label, 'lat': float(r.lat), 'lon': float(r.lon), 'n': int(r.n_mentions)} for r in dm.itertuples()}
    eid = lambda p: 'e:' + p[4:] if p.startswith('row:') else p          # the merges name editor rows e:N, the DicoTopo table row:N
    for art, g in mc.groupby('article_id'):
        pts = sorted({p for p in list(g['point_a']) + list(g['point_b'])}, key=lambda p: (not p.startswith('row:'), -info_dt[p]['n']))
        merges.append({'keep': eid(pts[0]), 'absorb': [eid(p) for p in pts[1:]], 'reading': 'both',
                       'reason': f'DicoTopo article {art} ({g.iloc[0]["headword"]}, INSEE {g.iloc[0]["insee"]}): same commune, {g["km_between"].max()} km apart at most',
                       'source': 'dicotopo', 'casebooks': False, 'new_label': None,
                       'info': {eid(p): {k: info_dt[p][k] for k in ('name', 'lat', 'lon')} for p in pts}})
    # integration check of 28 Oct: editor points of the same kind at the very same spot (3 decimals) whose names fold to one, or one holding the other as a
    # whole word, left over after the merges above and not on the keep-apart list. Read one by one: the three below are the same place.
    rows_b = json.loads(next(l for l in (R / 'page_kit' / 'v28' / 'build_b' / 'index.html').read_text('utf-8').split('\n')
                             if l.startswith('const D = '))[len('const D = '):].rstrip().rstrip(';'))['rows']
    for keep, other, new_label, why in (('e:6315', 'e:13319', 'Santander', 'Santander City (Wikidata Q23992498) and the register label of Santander: one point'),
                                         ('e:5461', 'e:12933', 'Bedworth', 'Bedworth, a descriptive spelling and the town: one point'),
                                         ('e:2557', 'e:5825', 'Pernambuco', 'Pernambuco, the captaincy label and the register label: one point'),
                                         # data check of 28 Oct (go/no-go): duplicates the exact-name rule missed because the register labels are in Portuguese, Spanish or the Casebooks spelling
                                         ('e:17', 'e:1085', None, 'Amsterdam and the Portuguese register label Amesterdão: one city, 0.5 km apart'),
                                         ('e:69', 'e:10804', None, 'Naples and the Spanish register label Nápoles: same city, same point to 3 decimals'),
                                         ('e:510', 'e:6919', 'Stowe', 'Stoe (Casebooks spelling) and Stowe parish: one place')):
        ids = [keep, other]
        merges.append({'keep': keep, 'absorb': [other], 'reading': 'both', 'reason': 'same kind, same point to 3 decimals, same name: ' + why, 'source': 'integration',
                       'casebooks': keep == 'e:510', 'new_label': new_label,
                       'info': {i: {'name': rows_b[int(i[2:])][0], 'lat': rows_b[int(i[2:])][1], 'lon': rows_b[int(i[2:])][2]} for i in ids}})
    used = set()
    for x in merges:
        used |= {x['keep'], *x['absorb']}
    for x in lf:
        used.add(x['point'])
    used |= {r.point_id for r in dm.itertuples()} | {'ml2:450'}
    model_keys = {i: prev[i] for i in sorted((u for u in used if u.startswith('ml2:')), key=lambda s: int(s[4:]))}
    labels = [{'point': x['point'], 'old': x['old'], 'new': x['new'], 'reason': x['reason']} for x in lf]
    out = {'note': "Fusions de points, corrections d'étiquettes et de QID du 28 oct. 2026 (décision 3), relues contradictoirement ; les fusions DicoTopo (décision 4) en font partie "
                   "(source dicotopo). e:N = ligne de D.rows de la page après l'audit des homonymes ; ml2:N = point de la couche modèle v3, retrouvé par model_keys (liste prev). "
                   "La lecture (reading) dit dans quelle lecture la fusion change quelque chose. "
                   "unlink : points du modèle qu'aucune passe automatique ne doit rattacher à un point éditeur (Belgia ne va pas à Bruxelles).",
           'unlink': ['ml2:450'], 'counts': {'merges': len(merges), 'labels': len(labels)}, 'qid_fixes': [{'point': 'e:41', 'old': 'Q31', 'new': 'Q239', 'reason': 'Brussels, not Belgium'}],
           'model_keys': model_keys, 'merges': merges, 'labels': labels}
    (CUR / 'place_merges.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n', 'utf-8')
    print('merges', len(merges), 'of which dicotopo', sum(1 for x in merges if x['source'] == 'dicotopo'), 'integration', sum(1 for x in merges if x['source'] == 'integration'), 'labels', len(labels), 'model keys', len(model_keys))

    # ---------------------------------------------------------------------------------------------------- DicoTopo names
    dm = dm.copy()
    dm['model_key'] = [json.dumps(prev[p]) if p.startswith('ml2:') else '' for p in dm['point_id']]
    dm['name'] = dm['label_final']
    cols = ['point_id', 'reading', 'model_key', 'current_label', 'name', 'headword', 'official_name_fr', 'article_id', 'old_id', 'insee', 'dep', 'qid', 'method',
            'matched_form', 'distance_km', 'notes', 'extra_forms_final', 'lat', 'lon']
    dm[cols].rename(columns={'extra_forms_final': 'extra_forms'}).to_csv(CUR / 'dicotopo_names.tsv', sep='\t', index=False)
    print('dicotopo names', len(dm))

    # ---------------------------------------------------------------------------------------------------- keep apart
    ka = pd.read_csv(V / 'duplicates' / 'keep_apart.tsv', sep='\t', keep_default_na=False)
    # Tuscolo / Tusculum (e:8508, e:9832) were kept apart before the review merged them (same place, 0.9 km): off the deny list
    ka = ka[~((ka['a'].isin(['e:8508', 'e:9832'])) & (ka['b'].isin(['e:8508', 'e:9832'])))]
    ka.to_csv(CUR / 'place_keep_apart.tsv', sep='\t', index=False)
    print('keep apart', len(ka))


if __name__ == '__main__':
    main()
