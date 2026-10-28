"""v2.2 through the build: kinds of towns, generic names with an id, multi-referent mentions, TGN points, the five-way kind of the globe."""
import json

import numpy as np
import pandas as pd

from ratlas import placekind as PK
from ratlas import reconcile_v2 as V
from ratlas.model import auth_to_arrow
from test_reconcile_gates import _tgn_dir2, _wd
from test_reconcile_v2 import _entity, _make, _mention

CLS = {'Q9001': 'municipality with town privileges in the Czech Republic', 'Q9002': 'big city', 'Q9003': 'geographic location', 'Q9004': 'castle',
       'Q9005': 'river', 'Q9006': 'province of Italy', 'Q9007': 'human settlement', 'Q9008': 'mountain range'}


def _scenario(tmp_path):
    A = 'colla'
    rows = [
        _mention(A, 'place', 'Prag', key='colla:prag', reg=True, auth={'wikidata': 'Q1085'}),
        _mention(A, 'place', 'Ballenstedt', key='colla:ball', reg=True, auth={'geonames': '2953060', 'wikidata': 'Q31901628'}),
        _mention(A, 'place', 'Mantua', key='colla:mantua', reg=True, auth={'geonames': '3173435', 'wikidata': 'Q16204'}),
        _mention(A, 'place', 'Heidelberger Schloss', key='colla:schloss', reg=True, auth={'wikidata': 'Q9100'}),
        _mention(A, 'place', 'Rhein', key='colla:rhein', reg=True, auth={'wikidata': 'Q9101'}),
        _mention(A, 'place', 'Alpen', key='colla:alpen', reg=True, auth={'wikidata': 'Q9102'}),
        # a common noun that carries an id: stays linked, flagged; its entity is named by the authority
        dict(_mention(A, 'place', 'alhier', key='colla:rue', reg=True, auth={'wikidata': 'Q9103'}), generic_with_id=True),
        dict(_mention(A, 'place', 'alhier', key='colla:rue', reg=True, auth={'wikidata': 'Q9103'}), generic_with_id=True),
        _mention(A, 'place', 'Aptera', key='colla:aptera', reg=True, auth={'tgn': '7011021'}),
        _mention(A, 'place', 'Lusitaniâ', key='colla:lusi', reg=True, auth={'tgn': '7030114'}),
        dict(_mention(A, 'person', 'Vatter', key='colla:vater', reg=True, auth={'wikidata': 'Q9104'}), generic_with_id=True),
    ]
    ents = [_entity(A, 'place', 'colla:prag', 'Prag'), _entity(A, 'place', 'colla:ball', 'Ballenstedt'), _entity(A, 'place', 'colla:mantua', 'Mantua'),
            _entity(A, 'place', 'colla:schloss', 'Heidelberger Schloss'), _entity(A, 'place', 'colla:rhein', 'Rhein'), _entity(A, 'place', 'colla:alpen', 'Alpen'),
            _entity(A, 'place', 'colla:rue', 'Rue'), _entity(A, 'place', 'colla:aptera', 'Aptera'), _entity(A, 'place', 'colla:lusi', 'Lusitaniâ'),
            _entity(A, 'person', 'colla:vater', 'Friedrich V.')]
    wd = [_wd('Q1085', {'en': 'Prague', 'de': 'Prag'}, classes=['Q9001', 'Q9002'], lat=50.088, lon=14.42),
          _wd('Q31901628', {'en': 'Ballenstedt', 'de': 'Ballenstedt'}, classes=['Q9003'], lat=51.729, lon=11.223),
          _wd('Q16204', {'en': 'Mantua', 'it': 'Mantova'}, classes=['Q9006'], lat=45.16, lon=10.79),
          _wd('Q9100', {'en': 'Heidelberg Castle', 'de': 'Heidelberger Schloss'}, classes=['Q9004'], lat=49.41, lon=8.72),
          _wd('Q9101', {'en': 'Rhine', 'de': 'Rhein'}, classes=['Q9005'], lat=51.0, lon=6.9),
          _wd('Q9102', {'en': 'Alps', 'de': 'Alpen'}, classes=['Q9008'], lat=46.5, lon=9.5),
          _wd('Q9103', {'en': 'Rue', 'fr': 'Rue'}, classes=['Q9007'], lat=50.27, lon=1.67),
          _wd('Q9104', {'en': 'Frederick V', 'de': 'Friedrich V.'}, classes=['Q5'], birth='+1596-08-26T00:00:00Z')]
    gn = [dict(gid='2953060', name='Ballenstedt', names='ballenstedt', lat=51.72, lon=11.23, pop=8064, cc='DE', fcode='PPL', fclass='P'),
          dict(gid='3173435', name='Mantova', names='mantua|mantova', lat=45.16, lon=10.79, pop=48000, cc='IT', fcode='PPLA2', fclass='P')]
    v2, rd = _make(tmp_path, rows, ents, extra_wd=wd, extra_gn=gn, extra_classes=CLS, extra_mapping={})
    return v2, rd


def test_towns_are_cities_generic_names_are_replaced_and_the_globe_has_five_kinds(tmp_path):
    v2, rd = _scenario(tmp_path)
    out = tmp_path / 'out'
    tgn_dir = _tgn_dir2(tmp_path)
    rep = V.build(v2, rd, out, tgn_dir=tgn_dir)
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    # Prague (classes of the Czech Republic), Ballenstedt (class "geographic location"), Mantua (item = the province, GeoNames = a town): cities
    assert geo.loc['colla:prag', 'place_kind'] == 'city' and geo.loc['colla:prag', 'coord_kind'] == 'point'
    assert geo.loc['colla:ball', 'place_kind'] == 'city' and geo.loc['colla:ball', 'coord_kind'] == 'point'
    assert geo.loc['colla:mantua', 'place_kind'] == 'city' and geo.loc['colla:mantua', 'kind_source'] == 'geonames'
    # TGN points that contradict the subject's hierarchy or the exonym table are not used; the ids stay accepted
    ap, lu = geo.loc['colla:aptera'], geo.loc['colla:lusi']
    assert ap['tgn_id'] == '7011021' and pd.isna(ap['tgn_lat']) and ap['coord_source'] == 'none' and lu['coord_source'] == 'none' and pd.isna(lu['tgn_lat'])
    tc = pd.read_csv(out / 'tgn_coord_check.tsv', sep='\t')
    assert set(tc['entity_key']) == {'colla:aptera', 'colla:lusi'} and tc.set_index('entity_key').loc['colla:aptera', 'tgn_coord_issue'].startswith('parent_far')
    assert tc.set_index('entity_key').loc['colla:lusi', 'tgn_coord_issue'].startswith('exonym_far')
    # the generic noun with an id: counted, linked, flagged, and not the name
    pg = pd.read_parquet(out / 'places_global.parquet').set_index('id')
    z = pg.loc['pl:Q9103']
    assert z['name'] == 'Rue' and not z['name_is_generic'] and bool(z['generic_with_id']) and z['n_generic_with_id'] == 2 and z['n_mentions'] == 2
    zl = geo.loc['colla:rue']
    assert zl['n_mentions'] == 2 and zl['qid'] == 'Q9103' and zl['generic_name'] and zl['n_generic_with_id'] == 2
    # persons: Vatter with a QID is named by Wikidata
    pe = pd.read_parquet(out / 'persons_global.parquet').set_index('id')
    v = pe.loc['pe:Q9104']
    assert v['name'] == 'Friedrich V.' and v['label'] == 'Frederick V' and not v['name_is_generic'] and bool(v['generic_with_id']) and v['n_generic_with_id'] == 1
    # the five kinds of the globe
    g = json.loads((out / 'globe_places.json').read_text())
    k = {p['id']: p['k'] for p in g['places']}
    assert k['pl:Q1085'] == 'c' and k['pl:Q31901628'] == 'c' and k['pl:Q16204'] == 'c'
    assert k['pl:Q9100'] == 'b' and k['pl:Q9101'] == 'n' and k['pl:Q9102'] == 'n' and k['pl:Q9103'] == 'c'
    assert set(k.values()) <= set(PK.KINDS)
    assert 'five-way' in g['header']['fields']['k'] and rep['globe']['by_kind']['c'] >= 4


def test_aggregate_global_never_prefers_a_common_noun_as_the_name():
    from ratlas import reconcile_build as B
    loc = pd.DataFrame({'global_id': ['g1', 'g1', 'g2', 'g2', 'g3'], 'scope': ['authority'] * 5, 'basis': ['qid'] * 5, 'coll_slug': ['a', 'b', 'a', 'b', 'a'],
                        'renascor_id': ['a', 'b', 'a', 'b', 'a'], 'name': ['Stadt', 'Magdeburg', 'Stadt', 'alhier', 'Zerbst'],
                        'variants': [[], [], [], [], []], 'tier': ['T1'] * 5, 'licence': ['L'] * 5, 'n_mentions': [10, 2, 7, 3, 1],
                        'generic_name': [True, False, True, True, False]})
    out = B.aggregate_global(loc, 'pl', ['n_mentions'])
    assert out.loc['g1', 'name'] == 'Magdeburg' and not out.loc['g1', 'name_is_generic']        # 10 mentions of "Stadt" against 2 of the name
    assert out.loc['g2', 'name'] == 'Stadt' and out.loc['g2', 'name_is_generic']               # only nouns: the commonest, flagged
    assert out.loc['g3', 'name'] == 'Zerbst' and not out.loc['g3', 'name_is_generic']


def test_placekind_five_kinds_from_the_sources():
    k = PK.kind5
    assert k('city', ['municipality with town privileges in the Czech Republic', 'big city']) == 'c'
    assert k('area', ['sovereign state']) == 'a' and k('area', ['island country']) == 'a'
    assert k('other', ['river']) == 'n' and k('area', ['island']) == 'n' and k(None, [], ['ISL']) == 'n' and k(None, [], ['MT']) == 'n'
    assert k('other', ['castle']) == 'b' and k(None, [], ['CSTL']) == 'b' and k(None, [], ['ST']) == 'b' and k(None, [], [], [], ['stre']) == 'b'
    assert k(None, [], ['PPLA2']) == 'c' and k(None, [], ['PCLI']) == 'a' and k(None, [], [], ['inhabited place']) == 'c' and k(None, [], [], ['nation']) == 'a'
    assert k('other') == 'u' and k(None) == 'u' and k('city') == 'c' and k('area') == 'a'
    assert k('other', ['island', 'municipality of Italy']) == 'n'                       # a tie goes to the more specific kind (island over municipality)
    assert k('city', ['city', 'castle']) == 'c'                                          # the atlas kind breaks a tie
    assert k(None, [], [], [], [], exonym_kind='n') == 'n'
    assert PK.note_kind('place_f_church') == 'b' and PK.note_kind('place_pa_settlement') == 'c' and PK.note_kind('Rive') == 'n' and PK.note_kind('place') is None
    assert PK.note_kind('eidgenössischer ort, kanton') == 'a'
    assert PK.name_kind('Kloster Zinna') == 'b' and PK.name_kind('The Thames') == 'n' and PK.name_kind('Ballenstedt(-Hoym), Amt') == 'a' and PK.name_kind('Hamburg') is None
    assert PK.name_kind('Dorchester on Thames') is None and PK.name_kind('Évora Monte') is None and PK.name_kind('Monte Somma') == 'n' and PK.name_kind('Thames River') == 'n'
    assert PK.name_kind('Holy See') is None and PK.name_kind('Le Thor') is None and PK.name_kind('Bodensee') == 'n' and PK.name_kind('Königsstraße') == 'b'
    assert k('city', [], [], [], [], None, ['St. Paul’s Cathedral']) == 'b' and k('city', [], [], [], [], None, ['Paris']) == 'c'
    assert PK.head_kind('château') == 'b' and PK.head_kind('Duchies of Silesia') == 'a' and PK.gn_kind('ADM3') == 'c' and PK.gn_kind('ADM1') == 'a'
    assert set(PK.KINDS) == set(PK.KIND_LABEL)


def test_a_register_country_with_a_tgn_inhabited_place_is_an_area(tmp_path):
    """v2.2: 'Hessen' typed country by the register, whose TGN id is an inhabited place of that name, was a city (kind) while its register point was dropped as a
    capital's; the register's type outranks the TGN type (not Wikidata's city)."""
    A = 'colla'
    rows = [_mention(A, 'place', 'Hessen', key='colla:hessen', reg=True, auth={'tgn': '7007070'}),
            _mention(A, 'place', 'Stuttgart', key='colla:stut', reg=True, auth={'tgn': '7007070'})]
    ents = [_entity(A, 'place', 'colla:hessen', 'Stuttgart', 48.78, 9.18, note_type='country'), _entity(A, 'place', 'colla:stut', 'Stuttgart', 48.78, 9.18)]
    v2, rd = _make(tmp_path, rows, ents)
    out = tmp_path / 'out'
    V.build(v2, rd, out, tgn_dir=_tgn_dir2(tmp_path))
    geo = pd.read_parquet(out / 'places_local_geo.parquet').set_index('entity_key')
    assert geo.loc['colla:hessen', 'place_kind'] == 'area' and geo.loc['colla:hessen', 'kind_source'] == 'register'
    assert geo.loc['colla:stut', 'place_kind'] == 'city' and geo.loc['colla:stut', 'kind_source'] == 'tgn'
