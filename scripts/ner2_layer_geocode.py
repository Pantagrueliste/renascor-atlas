"""NER layer v1, geocoding stage 2 (Hypatia, Slurm): geocode the distinct (language, form) pairs of the recommended spans with the v2.4 name-based geocoder.

    python scripts/ner2_layer_geocode.py --forms GEO/forms.parquet --doc-forms GEO/doc_forms.parquet --out GEO/gc

It reuses ``ratlas.reconcile_geocode`` unchanged (exonym table, polities, Latin names, GeoNames / TGN name lookup, propagation from the places that editors tagged
and placed, ranking by population, feature type, regional prior and the places of the same documents). The v2.4 tables are read, never written:

* anchors: the places of the v2.4 atlas that have register / authority coordinates (``out_v2_4/reconcile/pass2/places_local_geo.parquet``); the places of v2.4 that
  have none are not targets (n_mentions set to 0 in memory), so only the layer's forms are geocoded;
* one pseudo-edition per language (``nerlayer_en`` ... ``nerlayer_la``) in the language family of the atlas (regional prior: English texts lean to Britain and
  Ireland, Italian ones to Italy ...); each form has the documents it occurs in as context (pass 2 adds the layer's own pass-1 results to the document clouds);
* no tag hints: rivers, mountains, seas and other GeoNames classes H / T / L are looked up only through the exonym / polity tables (a river is never a point
  of the layer by name alone);
* no passages for the Latin homonym families (Lugdunum, Traiectum, Augusta ...): without a cue a family member is not placed.

Writes: ``gn_index_layer.parquet`` (the GeoNames / TGN index of the layer's keys, built once), ``gc_results.parquet`` (one row per form: gc_* columns of the
v2.4 geocoder, ``gc_reason`` for the forms that stay unplaced), ``gc_run.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ratlas import reconcile_geocode as RG
from ratlas import latin_geo as LG
from ratlas import placeclass as PCL
from ratlas import generic_terms as GTM

LANG_FAMILY = {'en': 'english', 'it': 'italian', 'fr': 'french', 'es': 'iberian', 'de': 'german', 'la': 'neo_latin'}


def _t(msg):
    print(f'[layer-gc {time.strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forms', type=Path, required=True)
    ap.add_argument('--doc-forms', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--v24', type=Path, default=Path.home() / 'renascor-atlas-data/out_v2_4')
    ap.add_argument('--rdir', type=Path, default=Path.home() / 'renascor-atlas-data/reconcile_v2_4')
    ap.add_argument('--gn-zip', type=Path, default=Path.home() / 'renascor-atlas-data/geonames/allCountries.zip')
    ap.add_argument('--tgn-dir', type=Path, default=Path.home() / 'renascor-atlas-data/tgn')
    ap.add_argument('--orbis', type=Path, default=Path.home() / 'renascor-atlas/data/orbis_latinus.tsv')
    ap.add_argument('--index-only', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the N most frequent forms')
    ap.add_argument('--ctx-map', type=Path, default=None, help='v3: parquet (lang, key, ctx): the atlas edition (coll_slug) that is the context of the form, "" = the pseudo-edition of its language')
    ap.add_argument('--doc-link', type=Path, default=None, help='v3: parquet (doc_id, slug, base): documents of the layer with a possible twin in the atlas (same edition, same file name)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    forms = pd.read_parquet(a.forms)
    forms = forms[forms['status'] == 'ok'].copy()
    gen = [GTM.place_generic(f) is not None for f in forms['form']]
    forms['status'] = np.where(gen, 'generic', 'ok')
    forms_all = forms
    forms = forms[forms['status'] == 'ok'].sort_values(['n', 'lang', 'key'], ascending=[False, True, True]).reset_index(drop=True)
    _t(f'{len(forms)} forms to geocode ({int(sum(gen))} generic nouns left out)')

    # --- index of the layer's keys
    idx = a.out / 'gn_index_layer.parquet'
    if not idx.exists():
        loc = a.out / 'layer_names.parquet'
        pq.write_table(pa.table({'name': forms['form'].tolist(), 'variants': pa.array([[] for _ in range(len(forms))], type=pa.list_(pa.string()))}), loc)
        extra = LG.OrbisLatinus.load(a.orbis).modern_names()
        RG.build_index(loc, a.gn_zip, idx, a.tgn_dir, extra)
        _t(f'index built ({time.time() - t0:.0f}s)')
    if a.index_only:
        return
    if a.limit:
        forms = forms.head(a.limit).copy()
    gn = RG.GN(pd.read_parquet(idx))

    # --- places: the v2.4 atlas as anchors, then the layer's forms
    geo = a.v24 / 'reconcile/pass2/places_local_geo.parquet'
    local = a.v24 / 'all/places_local.parquet'
    mentions = a.v24 / 'all/mentions.parquet'
    P0 = RG.load_places(geo, local)
    n0 = len(P0)
    _t(f'{n0} places of v2.4, {int(P0["lat"].notna().sum())} placed')
    captured = []
    if a.doc_link is not None:                                  # v3: keep the document codes of the atlas, to link the layer's documents to their twins
        _orig_codes = RG.doc_codes

        def _capture(m):
            c = _orig_codes(m)
            captured.append(pd.DataFrame({'coll_slug': m['coll_slug'].to_numpy(), 'source_path': m['source_path'].fillna('').to_numpy(), 'code': c.to_numpy()}))
            return c
        RG.doc_codes = _capture
    pairs0, hint0, docreg = RG.load_pairs(mentions, P0)
    RG.doc_codes = _orig_codes if a.doc_link is not None else RG.doc_codes
    aux = RG.load_mention_aux(mentions, P0)
    P0['n_mentions'] = np.where(P0['lat'].notna(), P0['n_mentions'], 0)          # the unplaced places of v2.4 are no targets
    for lg, fam in LANG_FAMILY.items():
        RG.FAMILY['nerlayer_' + lg] = fam
    P1 = pd.DataFrame({c: pd.Series([None] * len(forms), dtype=object) for c in P0.columns})
    P1['coll_slug'] = 'nerlayer_' + forms['lang']
    if a.ctx_map is not None:                                   # v3: the edition of the atlas that is the context of the form (regional prior of that edition)
        cm = pd.read_parquet(a.ctx_map).set_index(['lang', 'key'])['ctx']
        cx = cm.reindex(pd.MultiIndex.from_arrays([forms['lang'], forms['key']])).fillna('').to_numpy()
        P1['coll_slug'] = np.where(cx != '', cx, P1['coll_slug'].to_numpy())
    P1['entity_key'] = 'surf:' + forms['lang'] + ':' + forms['key']
    P1['name'] = forms['form'].to_numpy()
    P1['norm_name'] = forms['key'].to_numpy()
    P1['n_mentions'] = forms['n'].to_numpy()
    P1['n_in_period'] = forms['n'].to_numpy()
    P1['tier'] = 'T1'
    P1['coord_source'] = 'none'
    P1['lat'] = np.nan
    P1['lon'] = np.nan
    P1['key_type'] = 'surface'
    P1['variants'] = [[] for _ in range(len(forms))]
    P1['place_class'] = [PCL.classify(n, None, None) for n in forms['form']]
    P1['family'] = [RG.FAMILY.get(c, LANG_FAMILY[x]) if not c.startswith('nerlayer_') else LANG_FAMILY[x] for c, x in zip(P1['coll_slug'], forms['lang'])]
    P1['src0'] = 'none'
    for c in ('lat', 'lon'):
        P1[c] = P1[c].astype('float64')
    P1['n_mentions'] = P1['n_mentions'].astype('int64')
    P1['n_in_period'] = P1['n_in_period'].astype('int64')
    P = pd.concat([P0, P1], ignore_index=True)
    for c in ('lat', 'lon'):
        P[c] = P[c].astype('float64')
    # pairs of the layer: (document, form)
    df = pd.read_parquet(a.doc_forms)
    rowof = pd.Series(np.arange(len(forms)) + n0, index=pd.MultiIndex.from_arrays([forms['lang'], forms['key']]))
    r = rowof.reindex(pd.MultiIndex.from_arrays([df['lang'], df['key']])).to_numpy()
    df = df.assign(row=r)
    df = df[df['row'].notna()].copy()
    df['row'] = df['row'].astype('int64')
    base = int(pairs0['doc'].max()) + 1
    df['doc'] = pd.factorize(df['doc_id'])[0] + base
    if a.doc_link is not None and captured:                     # v3: a document of the layer that is also in the atlas takes the atlas document code
        cap = pd.concat(captured, ignore_index=True).drop_duplicates(['coll_slug', 'source_path'])
        cap['base'] = cap['source_path'].map(lambda s: str(s).replace('\\', '/').rsplit('/', 1)[-1])
        cap = cap.drop_duplicates(['coll_slug', 'base'], keep=False)           # a file name that two files share is not a link
        lk = pd.read_parquet(a.doc_link).merge(cap[['coll_slug', 'base', 'code']], left_on=['slug', 'base'], right_on=['coll_slug', 'base'], how='inner')
        twin = lk.set_index('doc_id')['code']
        t = df['doc_id'].map(twin)
        df['doc'] = np.where(t.notna(), t.fillna(-1), df['doc']).astype('int64')
        _t(f'{len(twin)} documents of the layer linked to their twin in the atlas ({int(t.notna().sum())} (document, form) rows)')
    pairs = pd.concat([pairs0[['doc', 'row']], df[['doc', 'row']].drop_duplicates()], ignore_index=True)
    hint1 = pd.DataFrame({'h': 0.0, 't': 0.0, 'l': 0.0, 'n_known': 2.0, 'n_latin': np.where(forms['lang'].to_numpy() == 'la', 2.0, 0.0)}, index=np.arange(len(forms)) + n0)
    hint = pd.concat([hint0, hint1]) if 'n_known' in hint0 else pd.concat([hint0.assign(n_known=0.0, n_latin=0.0), hint1])
    _t(f'{len(P)} places, {len(pairs)} pairs ({len(df)} of the layer) ({time.time() - t0:.0f}s)')
    res = {'orbis': LG.OrbisLatinus.load(a.orbis), 'wdlatin': LG.WikidataLatin.load(a.rdir / 'wd_entities.parquet'), 'mention_aux': aux}
    G = RG.Geocoder(P, gn, pairs, hint, docreg, res)
    _t(f'geocoder ready ({time.time() - t0:.0f}s)')
    out = RG.run_pipeline(G, np.array([], dtype=int))
    rows = RG.result_rows(G, out)
    ur = RG.unresolved_rows(G, out)
    rows = rows.merge(ur, how='outer', on=['coll_slug', 'entity_key'])
    mine = rows[rows['entity_key'].isin(set(P1['entity_key']))].copy()
    mine['lang'] = [e.split(':', 2)[1] for e in mine['entity_key']]
    mine['key'] = [e.split(':', 2)[2] for e in mine['entity_key']]
    mine.to_parquet(a.out / 'gc_results.parquet', index=False)
    conf = mine['gc_conf'].value_counts(dropna=False).to_dict()
    summ = {'forms_geocoded': int(len(forms)), 'forms_generic_left_out': int(sum(gen)), 'results': int(len(mine)), 'conf': {str(k): int(v) for k, v in conf.items()},
            'reasons': {str(k): int(v) for k, v in mine['gc_reason'].value_counts().items()}, 'seconds': round(time.time() - t0)}
    (a.out / 'gc_run.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False), 'utf-8')
    forms_all.to_parquet(a.out / 'forms_status.parquet', index=False)
    _t(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
