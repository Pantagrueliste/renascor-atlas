"""NER layer v3, geocoding context (CPU, Hypatia): which edition and which document of the atlas each form of the layer belongs to.

    python scripts/ner2_layer_v3_geo_ctx.py --geo ~/renascor-atlas-data/ner/layer_v3/geo --layer ~/renascor-atlas-data/ner/layer_v3/geo/layer

The geocoder (``ratlas.reconcile_geocode``, unchanged) ranks candidates with the placed places of the same edition (regional prior of the collection) and of the same document
(document cloud). The layer v1 had one pseudo-edition per language (``nerlayer_<lang>``). For the TEI editions the editors' places of the edition exist in the atlas v2.4, so:

* ``ctx_map.parquet`` (lang, key, ctx): the context edition of each form. A form takes the atlas edition (``COLL_SLUG``) that holds at least half of its mentions, when that edition
  has at least ``MIN_ANCHORS`` placed mentions in v2.4 and no hard regional box or unique-candidate path of its own (those rules were made for the editors' own places); otherwise it
  keeps the pseudo-edition of its language, as in v1.
* ``doc_link.parquet`` (doc_id, slug, base): the documents of the layer that may have a twin in the atlas (same edition, same file name). The geocoder links the twin: the editors'
  places of that very document become the document cloud of the layer's forms of that document.
* ``ctx_report.tsv``: forms and mentions by collection and by context edition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: collection of the layer -> edition (coll_slug) of the atlas v2.4 that holds the editors' places of the same edition
COLL_SLUG = {
    'BullingerLetters': 'bullinger', 'DTA': 'dta_1600', 'DTA_komplett': 'dta_1600', 'GerManC': 'germanc', 'CroALa': 'croala', 'PostScriptum': 'postscriptum',
    'BDPN_Novohispano': 'bdpn', 'Suriano_Correspondence': 'suriano', 'REED_English': 'reed', 'Controversia_Confessio': 'controversia_confessio', 'CERA': 'camena_cera',
    'CAMENA_CERA_Letters': 'camena_cera', 'Casebooks': 'casebooks', 'CORPUS17_French': 'corpus17_french', 'DiBiLit_German': 'dibilit_german', 'CorpusCorp': 'corpus_corporum',
    'Cavriana': 'cavriana', 'CKCC': 'ckcc_epistolarium',
}
MIN_ANCHORS = 1000


def choose_ctx(counts: pd.DataFrame, usable: set, share: float = 0.5) -> pd.Series:
    """counts: rows (lang, key, slug, n) = mentions of a form per atlas edition ('' = no edition). Returns Series indexed by (lang, key): the edition holding >= share of the
    form's mentions when it is in ``usable``, else ''."""
    tot = counts.groupby(['lang', 'key'])['n'].sum().rename('tot')
    c = counts[counts['slug'].isin(usable)].groupby(['lang', 'key', 'slug'])['n'].sum().reset_index()
    c = c.sort_values(['lang', 'key', 'n', 'slug'], ascending=[True, True, False, True]).drop_duplicates(['lang', 'key']).set_index(['lang', 'key'])
    c = c.join(tot, how='right')
    ok = (c['n'] / c['tot'] >= share) & c['slug'].notna()
    return c['slug'].where(ok, '').fillna('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geo', type=Path, required=True)
    ap.add_argument('--layer', type=Path, required=True)
    ap.add_argument('--v24', type=Path, default=Path.home() / 'renascor-atlas-data/out_v2_4')
    a = ap.parse_args()
    from ratlas import reconcile_geocode as RG
    P0 = RG.load_places(a.v24 / 'reconcile/pass2/places_local_geo.parquet', a.v24 / 'all/places_local.parquet')
    placed = P0[P0['lat'].notna()].groupby('coll_slug')['n_mentions'].sum()
    usable = {s for s in set(COLL_SLUG.values()) if placed.get(s, 0) >= MIN_ANCHORS and s not in RG.HARD_REGIONS and s not in RG.REGIONAL_UNIQUE}
    print('editions with anchors:', {s: int(placed.get(s, 0)) for s in sorted(set(COLL_SLUG.values()))})
    print('usable as context:', sorted(usable))
    docs = pd.read_parquet(a.layer / 'docs.parquet', columns=['doc_id', 'collection', 'member'])
    df = pd.read_parquet(a.geo / 'doc_forms.parquet')
    df = df.merge(docs[['doc_id', 'collection']], on='doc_id', how='left')
    df['slug'] = df['collection'].map(COLL_SLUG).fillna('')
    counts = df.groupby(['lang', 'key', 'slug'])['n'].sum().reset_index()
    ctx = choose_ctx(counts, usable).rename('ctx').reset_index()
    ctx.to_parquet(a.geo / 'ctx_map.parquet', index=False)
    docs['slug'] = docs['collection'].map(COLL_SLUG).fillna('')
    docs['base'] = docs['member'].fillna('').map(lambda s: str(s).replace('\\', '/').rsplit('/', 1)[-1])
    docs[docs['slug'] != ''][['doc_id', 'slug', 'base']].to_parquet(a.geo / 'doc_link.parquet', index=False)
    f = pd.read_parquet(a.geo / 'forms.parquet')[['lang', 'key', 'n']].merge(ctx, on=['lang', 'key'], how='left')
    rep = f.groupby(['lang', 'ctx']).agg(forms=('key', 'size'), mentions=('n', 'sum')).reset_index()
    rep.to_csv(a.geo / 'ctx_report.tsv', sep='\t', index=False)
    print(rep.to_string())


if __name__ == '__main__':
    main()
