"""v2.3 through the build and the cards: Casebooks points of low geocoding confidence, register points of areas, the qualifier's kind, exonym results joining the
item's cluster, the merge of split points, kind overrides, display names, the Latin source code."""
import json

import numpy as np
import pandas as pd

from ratlas import cards_globe as CG
from ratlas import placekind as PK
from ratlas import reconcile_build as B
from ratlas import reconcile_v2 as V
from test_reconcile_gates import _wd
from test_reconcile_v2 import _entity, _make, _mention


# ---- Casebooks ----------------------------------------------------------------------------------------------------------------------------------
def _cb_map(tmp_path, rows):
    cols = ['place_id', 'surfaces', 'directory_entry', 'entry_type', 'county', 'parish', 'country', 'lat', 'lon', 'confidence', 'method', 'n_mentions', 'score', 'spatial_p',
            'geo_method', 'geo_confidence', 'geo_radius_km', 'geonames_id', 'geonames_name', 'directory_path', 'alt_candidates']
    d = [dict.fromkeys(cols, '') | r for r in rows]
    p = tmp_path / 'casebooks_place_map.tsv'
    pd.DataFrame(d, columns=cols).to_csv(p, sep='\t', index=False)
    return p


def test_casebooks_point_with_low_geocoding_confidence_is_not_a_precise_point(tmp_path):
    """St George's Lane was put in Kentish Town by a guess of the directory (geocoding 'low'): the match is right, the point is not a point."""
    m = _cb_map(tmp_path, [dict(place_id='PLACE1', directory_entry="St George's Lane", county='London', lat='51.556', lon='-0.128', confidence='high', geo_confidence='low',
                                geo_radius_km='2.0'),
                           dict(place_id='PLACE2', directory_entry='Stondon', county='Bedfordshire', lat='52.0', lon='-0.3', confidence='high', geo_confidence='high',
                                geo_radius_km='2.0'),
                           dict(place_id='PLACE3', directory_entry='Bedfordshire', county='Bedfordshire', lat='52.1', lon='-0.4', confidence='medium', geo_confidence='low',
                                geo_radius_km='25.0')])
    pl = pd.DataFrame({'coll_slug': 'casebooks', 'entity_key': ['casebooks:PLACE1', 'casebooks:PLACE2', 'casebooks:PLACE3'], 'name': ['a', 'b', 'c'],
                       'coord_source_geo': 'none', 'geo_lat': np.nan, 'geo_lon': np.nan, 'coord_conf': None, 'coord_kind': None, 'coord_radius_km': np.nan,
                       'place_kind': None, 'n_in_period': [3, 5, 7], 'coord_source_pre': 'none'})
    rep = {}
    out = V.apply_casebooks(pl, m, rep).set_index('entity_key')
    assert not out.loc['casebooks:PLACE1', 'cb_used'] and out.loc['casebooks:PLACE1', 'coord_source_geo'] == 'none'          # stays a suggestion
    assert float(out.loc['casebooks:PLACE1', 'cb_lat']) == 51.556                                                         # ... which is kept
    assert out.loc['casebooks:PLACE2', 'cb_used'] and out.loc['casebooks:PLACE2', 'coord_kind'] == 'point'
    assert out.loc['casebooks:PLACE3', 'cb_used'] and out.loc['casebooks:PLACE3', 'coord_kind'] == 'area'                  # an area-level point (county median) is kept as an area
    assert rep['casebooks_directory']['geo_low_point_not_used'] == 1 and rep['casebooks_directory']['placed_distinct'] == 2


# ---- the build ---------------------------------------------------------------------------------------------------------------------------
def _scenario(tmp_path):
    A, Bc = 'colla', 'collb'
    rows = [_mention(A, 'place', 'Solothurn', key='colla:solo', reg=True, auth={'geonames': '2658561'}),
            _mention(A, 'place', 'Solothurn', key='colla:solo', reg=True, auth={'geonames': '2658561'}),
            _mention(A, 'place', 'Frankreich', key='colla:fr', reg=True, auth={'wikidata': 'Q142'}),
            _mention(Bc, 'place', 'Galli'), _mention(Bc, 'place', 'Galli'), _mention(Bc, 'place', 'France'),
            _mention(Bc, 'place', 'Lingen, Grafschaft'), _mention(Bc, 'place', 'Colmariam'), _mention(Bc, 'place', 'Colmariam', year=1650),
            _mention(A, 'place', 'Stadt Zerbst'), _mention(Bc, 'place', 'Anhalt-Zerbst, Fürstentum')]
    ents = [_entity(A, 'place', 'colla:solo', 'Solothurn', 47.2079, 7.5371, auth={'geonames': '2658561'}, note_type='region'),
            _entity(A, 'place', 'colla:fr', 'Frankreich', 46.0, 2.0, auth={'wikidata': 'Q142'})]
    wd = [_wd('Q142', {'en': 'France', 'de': 'Frankreich'}, classes=['Q6256'], lat=46.6, lon=2.5)]
    gn = [dict(gid='2658561', name='Canton of Solothurn', names='solothurn|canton of solothurn|kanton solothurn', lat=47.30, lon=7.65, pop=270000, cc='CH', fcode='ADM1', fclass='A')]
    v2, rd = _make(tmp_path, rows, ents, extra_wd=wd, extra_gn=gn)
    return v2, rd


def test_build_v23_areas_qualifier_exonym_cluster_kind_overrides_and_merged_points(tmp_path):
    v2, rd = _scenario(tmp_path)
    (rd / 'kind_overrides.tsv').write_text('# comment\ncoll_slug\tentity_key\tkind\tnote\ncolla\tcolla:solo\tn\ta test\n', encoding='utf-8')
    gc = pd.DataFrame({'coll_slug': ['collb', 'collb', 'collb', 'collb', 'colla', 'collb'],
                       'entity_key': ['surf:galli', 'surf:france', 'x', 'y', 'surf:stadt zerbst', 'surf:anhalt zerbst furstentum'],
                       'name': ['Galli', 'France', 'Lingen, Grafschaft', 'Colmariam', 'Stadt Zerbst', 'Anhalt-Zerbst, Fürstentum'],
                       'gc_lat': [46.6, 46.6, 52.52, 41.9, 51.966, 51.966], 'gc_lon': [2.5, 2.5, 7.32, 12.5, 12.085, 12.085],
                       'gc_source': ['name_exonym'] * 2 + ['name_propagation', 'name_latin', 'name_geonames', 'name_propagation'],
                       'gc_conf': ['medium'] * 2 + ['medium', 'medium', 'high', 'medium'],
                       'gc_radius_km': [350.0, 350.0, 5.0, 10.0, 5.0, 5.0],
                       'gc_cand': ['exonym:France'] * 2 + ['colla:colla:x', 'colla:colla:rome', 'gn:3', 'colla:surf:stadt zerbst'],
                       'gc_cand_name': ['France', 'France', 'Lingen', 'Roma', 'Zerbst', 'Zerbst'],
                       'gc_evidence': ['exonym table'] * 2 + ['x', 'Latin Colmariam', 'GeoNames 3 Zerbst (PPLA3, DE, pop 20000); key', 'copy'],
                       'gc_qid': [None] * 6})
    # the surface keys of the entities: only those that exist are used; build the keys from the table itself
    pl0 = pd.read_parquet(v2 / 'places_local.parquet')
    keys = {n: k for n, k in zip(pl0['name'], pl0['entity_key'])}
    gc['entity_key'] = [keys.get(n, k) for n, k in zip(gc['name'], gc['entity_key'])]
    gc['gc_cand'] = ['exonym:France'] * 2 + [f"colla:{keys['Solothurn']}", f"colla:{keys['Solothurn']}", 'gn:3', f"colla:{keys['Stadt Zerbst']}"]
    gcp = tmp_path / 'geocode_places.parquet'
    gc.to_parquet(gcp)
    out = tmp_path / 'out'
    rep = V.build(v2, rd, out, geocode=gcp)
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index(['coll_slug', 'entity_key'])
    g_ = lambda c, k: geo.loc[(c, k)]      # noqa: E731
    # 1. the register point of a 'region' that sits on the city is replaced by the authority's centroid (Solothurn canton)
    so = g_('colla', 'colla:solo')
    assert so['reg_area_replaced'] and so['coord_source'] == 'geonames' and abs(so['lat'] - 47.30) < 0.01 and so['coord_kind'] == 'area'
    assert rep['places_local'].get('rows') and 'kind_overrides' in rep and rep['kind_overrides']['matched'] == 1
    # 2. the qualifier sets the kind: a county, an area with the radius of the qualifier, whatever the head is
    li = g_('collb', keys['Lingen, Grafschaft'])
    assert li['place_kind'] == 'area' and li['kind_source'] == 'qualifier' and li['coord_kind'] == 'area' and li['coord_radius_km'] == 40.0
    assert g_('collb', keys['Anhalt-Zerbst, Fürstentum'])['coord_kind'] == 'area'
    assert rep['qualifier_kind']['area_by_qualifier'] >= 2
    # 3. exonym results join the item's cluster: Galli, France and Frankreich are ONE global place with the item Q142 (one count)
    pg = pd.read_parquet(out / 'places_global.parquet').set_index('id')
    assert g_('collb', keys['Galli'])['global_id'] == 'pl:Q142' and g_('collb', keys['France'])['global_id'] == 'pl:Q142' and g_('colla', 'colla:fr')['global_id'] == 'pl:Q142'
    assert g_('collb', keys['Galli'])['qid_exo'] == 'Q142' and pd.isna(g_('collb', keys['Galli'])['qid'])                      # the local's own accepted ids are untouched
    assert pg.loc['pl:Q142', 'n_in_period'] == 5 and rep['exonym_qid_attached']['by_item'] == {'Q142': 2}
    # 4. the Latin source is a free (name-based) source: it never overrides a register point and is counted
    ro = g_('collb', keys['Colmariam'])
    assert ro['coord_source'] == 'name_latin' and ro['coord_conf'] == 'medium'
    assert rep['places_local']['source_name_latin_distinct_in_period'] == 1
    # 5. the globe: one France point, source letters, kind override
    g = json.loads((out / 'globe_places.json').read_text())
    pts = {p['id']: p for p in g['places']}
    fr = [p for p in g['places'] if p['name'] in ('Frankreich', 'France')]
    assert len(fr) == 1 and fr[0]['n'] == 5 and fr[0]['k'] == 'a'
    assert pts[so['global_id']]['k'] == 'n'                                                                              # kind_overrides.tsv
    assert 'l=Latin name' in g['header']['fields']['s'] and any(p['s'] == 'l' for p in g['places'])
    # 6. the Zerbst case: the town and the principality that copied its point are one point of the globe
    zs = [p for p in g['places'] if abs(p['lat'] - 51.966) < 0.01]
    assert len(zs) == 1 and zs[0]['n'] == 2


def test_point_unions_same_candidate_geonames_cluster_propagation_exonym_and_display_name():
    ag = pd.DataFrame({
        'id': ['pl:Q1', 'pl:e:a:1', 'pl:e:b:2', 'pl:geonames:77', 'pl:e:c:3', 'pl:Q142', 'pl:e:d:4', 'pl:e:e:5', 'pl:e:f:6', 'pl:e:g:7'],
        'name': ['Ballenstedt', 'Ballenstedt(-Hoym), Amt', 'Utrecht', 'Offenburg', 'Offenbourg', 'Frankreich', 'France', 'Westminster Abbey', 'Westminster', 'Westminster Hall'],
        'label_en': ['Ballenstedt', None, None, 'Offenburg', None, 'France', 'France', None, 'Westminster', None],
        'label': ['Ballenstedt', None, None, 'Offenburg', None, 'France', 'France', None, 'Westminster', None],
        'label_fr': None, 'label_de': None, 'label_it': None,
        'qid': ['Q31901628', None, None, None, None, 'Q142', None, None, 'Q189960', None],
        'geonames': [None, None, None, '77', None, None, None, None, None, None],
        'lat': [51.729, 51.729, 52.09, 48.47, 48.47, 47.0, 46.6, 51.4988, 51.4995, 51.4996], 'lon': [11.22, 11.22, 5.12, 7.94, 7.94, 2.0, 2.5, -0.1277, -0.1333, -0.1255],
        'n_in_period': [10, 4, 5, 6, 2, 100, 40, 5, 9, 3],
        'place_kind': ['city', 'city', 'city', 'city', 'city', 'area', 'area', 'city', 'city', 'city'],
        'coord_cand': [None, 'colla:tagebuecher:ballenstedt', 'gn:555', None, 'gn:77', None, 'exonym:France', None, None, None],
        'coord_kind': ['point', 'point', 'point', 'point', 'point', 'area', 'area', 'point', 'point', 'point'],
        'coord_radius_km': [np.nan, np.nan, np.nan, np.nan, np.nan, 350.0, 350.0, np.nan, np.nan, np.nan]})
    pl = pd.DataFrame({'coll_slug': ['colla'], 'entity_key': ['tagebuecher:ballenstedt'], 'global_id': ['pl:Q1']})
    pairs = {tuple(sorted(p)) for p in V.point_unions(ag, pl)}
    assert ('pl:Q1', 'pl:e:a:1') in pairs                                    # copied from the town (propagation source), 0 km
    assert ('pl:e:c:3', 'pl:geonames:77') in pairs                           # a name-based place on GeoNames feature 77 and the cluster of that id
    assert ('pl:Q142', 'pl:e:d:4') in pairs                                  # the exonym 'France' and the item Q142: 58 km, inside the radius
    assert ('pl:e:e:5', 'pl:e:f:6') not in pairs and ('pl:e:f:6', 'pl:e:g:7') not in pairs      # the abbey and the city are not the same name
    gid = pd.Series(B.merge_points(ag, sorted(pairs)).to_numpy(), index=ag['id'])
    assert gid['pl:Q1'] == gid['pl:e:a:1'] and gid['pl:Q142'] == gid['pl:e:d:4'] and gid['pl:geonames:77'] == gid['pl:e:c:3']
    assert gid['pl:e:b:2'] != gid['pl:Q1']
    # same display name within 5 km: two places called Utrecht
    ag2 = pd.DataFrame({'id': ['x1', 'x2'], 'name': ['Utrecht', 'Utrecht'], 'label_en': [None, None], 'label': [None, None], 'qid': [None, None], 'geonames': [None, None],
                        'lat': [52.0910, 52.0912], 'lon': [5.1220, 5.1221], 'n_in_period': [3, 2], 'coord_cand': [None, None], 'coord_kind': ['point', 'point'],
                        'coord_radius_km': [np.nan, np.nan], 'label_fr': None, 'label_de': None, 'label_it': None})
    assert ('x1', 'x2') in {tuple(sorted(p)) for p in V.point_unions(ag2, pl)}


def test_exonym_items_are_verified_against_the_english_label():
    class A:
        wd = {'Q142': {'labels': {'en': 'France'}, 'missing': False}, 'Q29': {'labels': {'en': 'Not Spain'}, 'missing': False}, 'Q170072': {'labels': {'en': 'Dutch Republic'}}}
    pl = pd.DataFrame({'coll_slug': 'c', 'entity_key': ['a', 'b', 'c', 'd'], 'qid': [None, None, None, 'Q1'], 'n_in_period': 1,
                       'coord_source_geo': 'name_exonym', 'gc_cand': ['exonym:France', 'exonym:Spain', 'exonym:Dutch Republic', 'exonym:France'],
                       'gc_qid': [None, None, 'Q170072', None]})
    out = V.attach_exonym_qids(pl, A())
    assert out['qid_exo'].tolist()[0] == 'Q142' and out['qid_exo'].isna().tolist() == [False, True, False, True]    # Spain's item has another English label here: nothing attached
    assert out['qid_cluster'].tolist()[0] == 'Q142' and out['qid_cluster'].tolist()[2:] == ['Q170072', 'Q1']


# ---- cards: display names ---------------------------------------------------------------------------------------------------------------------
def test_display_names_register_name_before_surface_and_bot_labels():
    lg = pd.DataFrame({'entity_key': ['surf:marian', 'bullinger:l9', 'bullinger:l10'], 'name': ['Marian', 'Marignano', 'alhier'], 'n_in_period': [9, 4, 12]})
    assert CG.register_name(lg) == 'Marignano'                                 # the register's form, not the commonest surface or a noun
    assert CG.register_name(lg.iloc[[0]]) is None and CG.register_name(lg.iloc[[2]]) is None
    assert CG.BOT_LABEL.search('lsjbot Duisburg') and not CG.BOT_LABEL.search('Duisburg')
    assert CG.DUPLICATE_ITEMS['Q31976906'] == 'Q2100'
    assert CG.SOURCE_CODES['l'].startswith('Latin') and 'historical' in CG.SOURCE_CODES['e']


def test_placekind_rules_of_v23():
    assert PK.name_kind('Rorbach') is None and PK.name_kind('Offenbach') is None and PK.name_kind('Bodensee') == 'n'            # villages in -bach
    assert PK.name_kind('Parish of St. Martin (Ludgate)') == 'b' and PK.name_kind('Parish of Weston') is None
    assert PK.kind5('city', [], [], [], [], None, ['Rorbach']) == 'c'


# ---- letters_docs.json is part of the cards build ---------------------------------------------------------------------------------------------
def test_cards_build_writes_letters_docs_and_the_letters_subcommand_rebuilds_them(tmp_path):
    import argparse
    import inspect
    src = inspect.getsource(CG.build)
    assert 'with_docs=True' in src and "'letters_docs.json'" in src and "'letters_coverage.tsv'" in src and "'letters.json'" in src          # the full build writes all three
    fin = tmp_path / 'final'
    fin.mkdir()
    (fin / 'globe_places.json').write_text(json.dumps({'places': [{'id': 'p0'}, {'id': 'p1'}], 'header': {'collections': []}}))
    pd.DataFrame({'id': ['g0', 'g1'], 'globe_id': ['p0', 'p1']}).to_parquet(fin / 'places_global.parquet')
    pd.DataFrame({'coll_slug': 'suriano', 'entity_key': ['suriano:hague', 'suriano:venice'], 'global_id': ['g0', 'g1']}).to_parquet(fin / 'places_local_geo.parquet')
    rows = []
    for d, (a, b) in enumerate([('hague', 'venice'), ('hague', 'venice')]):
        for role, k in (('sent_from', a), ('sent_to', b)):
            rows.append(dict(coll_slug='suriano', kind='place', canonical=True, excluded=False, local_key=f'suriano:{k}', surface=k, in_period=True,
                             source_path=f'l{d}.xml', doc_id=f'l{d}', role=role, doc_year=1616))
    pd.DataFrame(rows).to_parquet(tmp_path / 'mentions.parquet')
    from ratlas.docs_table import doc_key
    dk = [doc_key('suriano', f'l{d}.xml', f'l{d}') for d in range(2)]
    pd.DataFrame({'doc_key': dk, 'title': ['Letter 1', 'Letter 2'], 'date': ['1616-08-16', '1616-08-17'], 'year_from': [1616, 1616], 'author': ['Suriano', 'Suriano'],
                  'sender': ['Suriano', 'Suriano'], 'addressee': ['Doge', 'Doge'], 'doc_type': 'letter', 'url': ['u1', 'u2'], 'coll_slug': 'suriano',
                  'in_period': True}).to_parquet(tmp_path / 'documents.parquet')
    (tmp_path / 'core.json').write_text(json.dumps({'slugs': ['suriano']}))
    out = tmp_path / 'out'
    out.mkdir()
    a = argparse.Namespace(final=fin, mentions=tmp_path / 'mentions.parquet', documents=tmp_path / 'documents.parquet', core=tmp_path / 'core.json', check=None, out=out)
    rep = CG.letters_only(a)
    assert rep['edges'] == 1 and rep['letters_in_edges'] == 2 and rep['without_document_row'] == 0
    edges = json.loads((out / 'letters.json').read_text())
    docs = json.loads((out / 'letters_docs.json').read_text())
    assert edges[0][:3] == [0, 1, 2] and [r[0] for r in docs['0']] == ['Letter 1', 'Letter 2'] and docs['0'][0][3:6] == ['Suriano', 'Doge', 0]


def test_the_page_is_told_which_codes_are_located_by_name():
    assert set(CG.NAME_BASED_CODES) == {'p', 'n', 'e', 'l'} and set(CG.NAME_BASED_CODES) <= set(CG.SOURCE_CODES)


def test_polity_and_exonym_items_exist_in_the_wikidata_table_of_the_build():
    """Real-data test (Hypatia: RATLAS_AUDIT_RDIR points to the reconcile directory): every Wikidata item of the reviewed polity table and of EXONYM_QID is in the local table of
    2026-10-17, is not missing, and the countries of EXONYM_QID carry the English label of the exonym entry."""
    import os
    import pytest
    from pathlib import Path
    from ratlas import polities as POL
    r = os.environ.get('RATLAS_AUDIT_RDIR')
    if not r or not (Path(r) / 'wd_entities.parquet').exists():
        pytest.skip('needs the reconcile directory of the build')
    wd = pd.read_parquet(Path(r) / 'wd_entities.parquet', columns=['qid', 'labels', 'missing']).set_index('qid')
    for p in POL.POLITIES:
        if p.qid:
            assert p.qid in wd.index and not wd.at[p.qid, 'missing'], (p.label, p.qid)
    for lab, q in V.EXONYM_QID.items():
        assert q in wd.index, (lab, q)
        if lab not in ('Bohemia', 'Pomerania', 'Mecklenburg', 'Silesia', 'Hesse'):
            assert json.loads(wd.at[q, 'labels']).get('en', '').lower() == lab.lower(), (lab, q)


def test_casebooks_name_based_guess_far_from_an_unused_directory_point_is_dropped(tmp_path):
    m = _cb_map(tmp_path, [dict(place_id='PLACE1', directory_entry='Charterhouse', county='London', lat='51.52', lon='-0.10', confidence='high', geo_confidence='low', geo_radius_km='2.0'),
                           dict(place_id='PLACE2', directory_entry='Hanslope', county='Buckinghamshire', lat='52.1', lon='-0.8', confidence='high', geo_confidence='low', geo_radius_km='2.0')])
    pl = pd.DataFrame({'coll_slug': 'casebooks', 'entity_key': ['casebooks:PLACE1', 'casebooks:PLACE2'], 'name': ['a', 'b'], 'coord_source_geo': 'name_geonames',
                       'geo_lat': [51.1, 52.11], 'geo_lon': [-2.7, -0.79], 'coord_conf': 'medium', 'coord_kind': 'point', 'coord_radius_km': np.nan, 'place_kind': None,
                       'n_in_period': [3, 5], 'coord_source_pre': 'none', 'gc_reason': None})
    rep = {}
    out = V.apply_casebooks(pl, m, rep).set_index('entity_key')
    assert out.loc['casebooks:PLACE1', 'coord_source_geo'] == 'none' and out.loc['casebooks:PLACE1', 'gc_reason'] == 'casebooks_directory_conflict'    # 120 km from the entry
    assert out.loc['casebooks:PLACE2', 'coord_source_geo'] == 'name_geonames'                                                                           # next to it: corroborated
    assert rep['casebooks_directory']['name_based_point_dropped_against_directory'] == 1
