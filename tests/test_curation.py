"""Homonym audit (ratlas.curation): the fixes file, the guard rules and the regression on the places the audit corrected.

The tests that read the atlas tables are skipped when ``RQ/derived/atlas/v2.4`` is not on this machine."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratlas import curation as C

ATLAS = Path(__file__).resolve().parents[2] / 'RQ' / 'derived' / 'atlas' / 'v2.4'
needs_atlas = pytest.mark.skipif(not (ATLAS / 'places_local_geo.parquet').exists(), reason='atlas v2.4 tables not here')


def frame(rows):
    """A norm_frame from (coll, key, name, lat, lon, source, n[, qid, gc_cand]) tuples."""
    d = []
    for r in rows:
        r = list(r) + [None] * (9 - len(r))
        d.append(dict(coll_slug=r[0], entity_key=r[1], name=r[2], lat=r[3], lon=r[4], coord_source=r[5], n_mentions=r[6], qid=r[7], gc_cand=r[8]))
    return C.norm_frame(pd.DataFrame(d))


EMPTY = pd.DataFrame(columns=['coll_slug', 'entity_key', 'name', 'action', 'lat', 'lon', 'kind', 'qid', 'reason_fr', 'origin'])


def test_files_are_well_formed():
    f = C.load_fixes()
    assert len(f) == 160
    assert f['action'].value_counts().to_dict() == {'move': 75, 'unplace': 39, 'keep': 35, 'not_a_place': 11}
    assert f['reason_fr'].str.len().min() > 10
    mv = f[f['action'] == 'move']
    assert mv['kind'].isin(list(C.KINDS)).all() and mv['lat'].between(-90, 90).all()
    a = C.load_ancient()
    assert a['sodom'] == ['LEV'] and a['atlantis'] == ['NONE'] and 'sion' not in a and 'tabor' not in a


def test_zone_and_regions():
    assert C.zone(37.9, -81.8) == 'americas' and C.zone(23.75, 89.6) == 'asia' and C.zone(0.4, 9.4) == 'africa_s'
    assert C.zone(31.2, 35.5) == 'old' and C.zone(-33.6, 18.5) == 'africa_s' and C.zone(-20.3, 57.6) == 'other'
    assert C.in_regions(31.2, 35.5, ['LEV']) and not C.in_regions(50.1, 14.4, ['GR']) and not C.in_regions(31.2, 35.5, ['NONE'])


def test_rule_ancient():
    n = frame([('bullinger', 'b:1', 'Sodom', 37.9165, -81.8332, 'register', 4),        # West Virginia
               ('x', 'x:1', 'Sodoma', 31.2, 35.5, 'wikidata', 3),                     # the Dead Sea: stays
               ('bullinger', 'b:2', 'Troja', 50.119, 14.422, 'register', 6),          # Prague
               ('camena', 'surf:atlantis', 'Atlantis', -33.57, 18.48, 'name_geonames', 1),
               ('y', 'y:1', 'Troja', 39.96, 26.24, 'wikidata', 9)])                   # Hisarlik: stays
    hit = C.rule_ancient(n, C.load_ancient())
    assert list(hit) == [True, False, True, True, False]
    p = C.plan(n, EMPTY, C.load_ancient())
    assert set(p['entity_key']) == {'b:1', 'b:2', 'surf:atlantis'} and set(p['why']) == {'rule:ancient'} and set(p['action']) == {'unplace'}


def test_keep_entry_protects_a_real_homonym():
    n = frame([('ckcc', 'surf:troye', 'Troye', 48.3, 4.08, 'name_propagation', 2)])       # Troyes (Champagne), read in context
    keep = pd.DataFrame([dict(coll_slug='ckcc', entity_key='surf:troye', name='Troye', action='keep', lat=np.nan, lon=np.nan, kind=None,
                              qid=None, reason_fr='Troyes en Champagne, lu.', origin='rule_checked')])
    assert len(C.plan(n, EMPTY, C.load_ancient())) == 1
    assert len(C.plan(n, keep, C.load_ancient())) == 0


def test_rule_far_homonym():
    rows = [('ed', f'ed:{i}', f'Town{i}', 47.0 + i * 0.01, 8.0, 'register', 50) for i in range(30)]    # an edition of the Swiss lands
    rows += [('ed', 'ed:alcala', 'Alcalá', 4.67, -75.78, 'register', 1),                             # Alcalá in Colombia
             ('ed', 'ed:lima', 'Lima', -12.05, -77.04, 'register', 1),                                 # no Old World Lima in the atlas: stays
             ('other', 'o:1', 'Alcalá', 40.48, -3.36, 'wikidata', 5)]                                  # Alcalá de Henares, elsewhere
    n = frame(rows)
    hit = C.rule_far_homonym(n)
    assert n.loc[hit, 'entity_key'].tolist() == ['ed:alcala']


def test_rule_propagated():
    n = frame([('bullinger', 'bullinger:l2382', 'Bethulia', 23.75, 89.63, 'register', 2),
               ('sandrart', 'sandrart:place-1198', 'Bethulia', 23.75, 89.63, 'name_propagation', 3, None, 'bullinger:bullinger:l2382'),
               ('z', 'z:1', 'Copy of a copy', 23.75, 89.63, 'name_propagation', 1, None, 'sandrart:sandrart:place-1198')])
    fx = pd.DataFrame([dict(coll_slug='bullinger', entity_key='bullinger:l2382', name='Bethulia', action='unplace', lat=np.nan, lon=np.nan, kind=None,
                            qid=None, reason_fr='Béthulie de Judith.', origin='verdict')])
    p = C.plan(n, fx, {})
    assert p.set_index('entity_key')['why'].to_dict() == {'bullinger:l2382': 'verdict', 'sandrart:place-1198': 'rule:propagated', 'z:1': 'rule:propagated'}


def test_supported():
    fx = dict(qid='Q170390', kind='a', lat=53.14, lon=8.22)
    assert C.supported(fx, 'Q170390', None, None, None, None) == 'qid'
    fx = dict(qid=None, kind='c', lat=47.1833, lon=11.4167)
    assert C.supported(fx, None, np.nan, np.nan, 47.17, 11.42) == 'tgn'                     # Schönberg: the edition's TGN id confirms it
    assert C.supported(fx, None, 48.73, 9.33, np.nan, np.nan) is None


def test_apply_reconcile(tmp_path):
    import json
    ents = [dict(coll_slug='b', entity_key='b:sodom', name='Sodom', action='unplace', reason_fr='Sodome biblique, lu.', origin='verdict'),
            dict(coll_slug='b', entity_key='b:tunis', name='Tunis', action='move', lat=36.8065, lon=10.1815, kind='c', qid='Q3572',
                 reason_fr='Tunis, la ville, lu.', origin='verdict'),
            dict(coll_slug='d', entity_key='d:schoen', name='Schönberg', action='move', lat=47.1833, lon=11.4167, kind='c',
                 reason_fr='Tyrol, lu dans le texte.', origin='verdict')]
    f = tmp_path / 'fixes.json'
    f.write_text(json.dumps({'entries': ents}), 'utf-8')
    pl = pd.DataFrame({
        'coll_slug': ['b', 'b', 'd', 'e'], 'entity_key': ['b:sodom', 'b:tunis', 'd:schoen', 'e:zurich'], 'name': ['Sodom', 'Tunis', 'Schönberg', 'Zürich'],
        'n_mentions': [4, 7, 2, 100], 'geo_lat': [37.9165, 33.84, 48.73, 47.37], 'geo_lon': [-81.83, 9.40, 9.33, 8.54],
        'coord_source_geo': ['register'] * 4, 'coord_kind': ['point'] * 4, 'place_kind': ['city'] * 4, 'coord_conf': ['authority'] * 4,
        'coord_radius_km': [np.nan] * 4, 'coord_source_pre': ['register'] * 4,
        'qid': ['Q1', None, None, 'Q72'], 'qid_cluster': ['Q1', None, None, 'Q72'], 'id_geonames': ['123', None, None, '2657896'],
        'has_reg': [True] * 4, 'has_wd': [False] * 4, 'has_gn': [False] * 4, 'has_tgn': [False, False, False, False],
        'auth_lat': [np.nan] * 4, 'auth_lon': [np.nan] * 4, 'auth_source': [None] * 4,
        'tgn_lat': [np.nan, np.nan, 47.17, np.nan], 'tgn_lon': [np.nan, np.nan, 11.42, np.nan], 'gc_cand': [None] * 4, 'gc_evidence': [None] * 4,
        'place_class': [None] * 4, 'identity_dropped': [False] * 4, 'kind5_override': [None] * 4})
    rep = {}
    out = C.apply_reconcile(pl, rep, f).set_index('entity_key')
    s = out.loc['b:sodom']
    assert s['coord_source_geo'] == 'none' and np.isnan(s['geo_lat']) and pd.isna(s['qid']) and pd.isna(s['id_geonames']) and not s['has_reg']
    t = out.loc['b:tunis']
    assert t['coord_source_geo'] == 'curated' and t['qid_cluster'] == 'Q3572' and t['coord_source_pre'] == 'none' and (t['geo_lat'], t['geo_lon']) == (36.8065, 10.1815)
    k = out.loc['d:schoen']
    assert k['coord_source_geo'] == 'tgn' and k['has_tgn'] and (k['tgn_lat'], k['tgn_lon']) == (47.1833, 11.4167)
    z = out.loc['e:zurich']
    assert z['coord_source_geo'] == 'register' and pd.isna(z['curation'])
    assert rep['curation'] == {'verdict:move': 2, 'verdict:unplace': 1}


@needs_atlas
def test_entries_exist_and_qids_fit():
    geo = pd.read_parquet(ATLAS / 'places_local_geo.parquet', columns=['coll_slug', 'entity_key'])
    have = set(zip(geo['coll_slug'], geo['entity_key']))
    f = C.load_fixes()
    missing = [k for k in zip(f['coll_slug'], f['entity_key']) if k not in have]
    assert not missing, missing[:5]
    pg = pd.read_parquet(ATLAS / 'places_global.parquet', columns=['qid', 'lat', 'lon']).dropna().drop_duplicates('qid').set_index('qid')
    for r in f[f['qid'].notna()].itertuples():
        if r.qid in pg.index:                       # a QID of an entry joins its cluster: it must be the place of the entry (Nîmes is Q42807)
            d = float(C.haversine_km(r.lat, r.lon, pg.at[r.qid, 'lat'], pg.at[r.qid, 'lon']))
            assert d < (300 if r.kind == 'a' else 50), (r.entity_key, r.qid, d)


@needs_atlas
def test_regression_audited_places_off_their_homonym():
    geo = pd.read_parquet(ATLAS / 'places_local_geo.parquet')
    n = C.norm_frame(geo)
    p = C.plan(n)
    f = C.load_fixes()
    todo = f[f['action'] != 'keep']
    got = p.set_index(['coll_slug', 'entity_key'])['action']
    for r in todo.itertuples():
        assert got.get((r.coll_slug, r.entity_key)) == r.action, (r.coll_slug, r.entity_key)
    after = C.apply_plan_frame(n, p).set_index(['coll_slug', 'entity_key'])
    assert np.isnan(after.at[('bullinger', 'bullinger:l2202'), 'lat'])                               # Sodom (West Virginia)
    assert np.isnan(after.at[('bullinger', 'bullinger:l2382'), 'lat'])                               # Bethulia (Bangladesh)
    assert after.at[('bullinger', 'bullinger:l2219'), 'lat'] == pytest.approx(39.9575)                # Troja: Troy, not Prague
    strict = p[p['strict']].set_index('entity_key')['strict_via'].to_dict()
    assert strict == {'hainhofer_reisen:oldenburg': 'qid', 'selbstzeugnisse_hab_diarium:Schoenberg_im_Stubaital': 'tgn',
                      'sozinianische_briefwechsel:ed_rl1_kf5_bnb': 'qid'}


@needs_atlas
def test_check_after_the_fixes_has_no_open_rule_hit():
    geo = pd.read_parquet(ATLAS / 'places_local_geo.parquet')
    t = C.check(C.norm_frame(geo))
    open_ = t[t['check'].isin(['ancient', 'far_homonym']) & ~t['covered']]
    assert open_.empty, open_[['coll_slug', 'entity_key', 'name']].values.tolist()[:5]
