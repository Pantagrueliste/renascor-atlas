"""Regression tests for the bugs found by the verification of the Atlas v1 tables (2026-10-16)."""
import numpy as np
import pandas as pd
import pytest
from lxml import etree

from ratlas import common as C
from ratlas import helpers_batch1 as H
from ratlas import merge as M
from ratlas import reconcile_build as B
from ratlas import reconcile_namecheck as N
from ratlas.collections import get


def test_parse_geo_leading_dot_decimals():
    # selbstzeugnisse_hab_diarium: '51.485 -.135' was read as (51.485, 135.0) and Westminster ended up in Siberia
    assert C._parse_geo('51.485 -.135') == (51.485, -0.135)
    assert C._parse_geo('51.333333 .75') == (51.333333, 0.75)
    assert C._parse_geo('52.37, 4.89') == (52.37, 4.89)
    assert C._parse_geo('48,85 2,35') == (48.85, 2.35)
    assert C._parse_geo('15.0') is None


def test_parse_coord_hemisphere_after_the_numbers_and_oeste():
    # h15corpus: '41°53′35″N 12°28′58″E' fell through to the decimal parser: (41.0, 53.0)
    lat, lon = H.parse_coord('41°53′35″N 12°28′58″E')
    assert abs(lat - 41.893056) < 1e-5 and abs(lon - 12.482778) < 1e-5
    lat, lon = H.parse_coord('N41°38\'53.84" O0°52\'58.91"')           # Spanish Oeste = west
    assert abs(lat - 41.648289) < 1e-5 and abs(lon + 0.883031) < 1e-5
    lat, lon = H.parse_coord('N45°27\'51.37" E9°11\'22.24"')
    assert abs(lat - 45.464269) < 1e-5 and abs(lon - 9.189511) < 1e-5
    assert H.parse_coord('41.65°, -0.883333°') == (41.65, -0.883333)


def test_choice_with_abbr_expan_reg_keeps_one_reading():
    # postscriptum: <abbr>d</abbr><expan>de</expan><reg>De</reg> gave the surface 'dDe'
    xml = ('<placeName><w><choice><abbr>d</abbr><expan>de</expan><reg>De</reg></choice></w> '
           '<w><choice><abbr>flça</abbr><expan>florença</expan><reg>Florença</reg></choice></w></placeName>')
    assert C.surface_text(etree.fromstring(xml)) == 'd flça'
    xml2 = '<placeName><choice><orig>Bernn</orig><reg>Bern</reg></choice></placeName>'
    assert C.surface_text(etree.fromstring(xml2)) == 'Bernn'
    xml3 = '<placeName><choice><reg>Bern</reg><sic>Bernn</sic></choice></placeName>'           # reg alone stays
    assert 'Bern' in C.surface_text(etree.fromstring(xml3))


def test_norm_drops_a_leading_of():
    assert M._norm('of Ashbey') == 'ashbey'
    assert M._norm('Of Ashbey') == 'ashbey'
    assert M._norm('of') == 'of' and M._norm('Offenbach') == 'offenbach'


SYN = """<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="X"><teiHeader><fileDesc><titleStmt><title>t</title></titleStmt>
<sourceDesc><p>Source: <name ref="mol:JONS1">Jonson, Ben</name>. London: <name ref="mol:BEAL4">John Beale</name>. <name ref="mol:HOLM3">Martin Holmes</name>
downloaded it at the <name type="org" ref="mol:UVIC3">University of Victoria</name>. <date when-custom="1615" datingMethod="mol:julianSic">1615</date></p></sourceDesc></fileDesc>
<profileDesc><textClass><catRef scheme="mdt:molDocumentTypes" target="mdt:mdtPrimarySourceLibraryMayoral"/></textClass></profileDesc></teiHeader>
<text><body><p><name ref="mol:HENR4">Henry</name> at <ref target="mol:CAND1">Candlewick</ref></p></body></text></TEI>"""


def test_moeml_drops_modern_staff_in_the_header_only():
    col = get('map_of_early_modern_london')
    ctx = col.make_context()
    rec = C.FileRecord(path='m/files/original/PRIM1.xml', coll='x', data=SYN.encode('utf8'))
    ctx.all_files = [rec]
    col.prepare(ctx)
    ms, _ = col.extract(rec, ctx)
    got = [(m.zone, m.surface) for m in ms]
    assert ('hdr', 'Jonson, Ben') in got and ('hdr', 'John Beale') in got          # historical authors / printers stay
    assert not any(s in ('Martin Holmes', 'University of Victoria') for _, s in got)
    assert ('txt', 'Henry') in got


def test_arbitrate_keeps_the_coordinate_that_agrees_with_the_name():
    pl = pd.DataFrame({
        'coll_slug': ['c'] * 4, 'entity_key': ['bordeaux', 'velletri', 'rhine', 'tie'],
        'qid': ['Q1204', 'Q189236', 'Q584', 'Q9'], 'qid_via': ['tgn'] * 4, 'id_tgn': ['7007251', 'x', 'y', 'z'],
        'has_reg': [True] * 4, 'has_wd': [True] * 4, 'has_gn': [False] * 4,
        'coord_conflict': [True] * 4,
    })
    nc = pd.DataFrame({'coll_slug': ['c'] * 4, 'entity_key': ['bordeaux', 'velletri', 'rhine', 'tie'],
                       'd_reg': [2.0, 721.0, 5.0, 900.0], 'd_auth': [6900.0, 3.0, 5.0, 800.0]}
                      ).set_index(['coll_slug', 'entity_key'])
    out = B.arbitrate(pl.copy(), nc)
    r = out.set_index('entity_key')
    assert r.at['bordeaux', 'arbitration'] == 'register' and not r.at['bordeaux', 'has_wd'] and pd.isna(r.at['bordeaux', 'qid'])
    assert pd.isna(r.at['bordeaux', 'id_tgn'])                      # no way back into the Illinois cluster
    assert r.at['velletri', 'arbitration'] == 'authority' and not r.at['velletri', 'has_reg'] and r.at['velletri', 'has_wd']
    assert pd.isna(r.at['rhine', 'arbitration']) and r.at['rhine', 'has_reg'] and r.at['rhine', 'has_wd']      # both agree: kept
    assert pd.isna(r.at['tie', 'arbitration'])
    same = B.arbitrate(pl.copy(), None)                                  # without a namecheck file nothing changes
    assert same['has_wd'].all() and same['arbitration'].isna().all()


def test_resolve_qids_prefers_the_item_that_is_not_a_duplicate():
    votes = pd.DataFrame({'coll_slug': ['c'], 'kind': ['place'], 'entity_key': ['k'], 'scheme': ['geonames'],
                          'id': ['1'], 'n': [3], 'n_ip': [3]})
    mapping = pd.DataFrame({'scheme': ['geonames', 'geonames'], 'id': ['1', '1'], 'qid': ['Q5', 'Q99'], 'k': [2, 2]})
    assert B.resolve_qids(votes, mapping, set(), 'place').iloc[0]['qid'] == 'Q5'          # tie -> lower number
    assert B.resolve_qids(votes, mapping, set(), 'place', {'Q5'}).iloc[0]['qid'] == 'Q99'
    only = pd.DataFrame({'scheme': ['geonames'], 'id': ['1'], 'qid': ['Q5'], 'k': [1]})
    assert B.resolve_qids(votes, only, set(), 'place', {'Q5'}).iloc[0]['qid'] == 'Q5'     # sole candidate is kept


def _pg(rows):
    cols = dict(name=None, label=None, label_en=None, label_fr=None, label_de=None, label_it=None, gn_name=None)
    return pd.DataFrame([{**cols, **r} for r in rows])


def test_merge_points_joins_same_name_within_20_km_only():
    pg = _pg([
        dict(id='pl:Q727', name='Amsterdam', label='Amsterdam', lat=52.3667, lon=4.8833, n_in_period=631),
        dict(id='pl:e:ckcc:1', name='Amsterdam', label='Amsterdam', lat=52.374, lon=4.891, n_in_period=2325),
        dict(id='pl:Q1794', name='Frankfurt (Main)', label='Frankfurt', label_de='Frankfurt am Main', lat=50.11, lon=8.68, n_in_period=475),
        dict(id='pl:Q4024', name='Frankfurt an der Oder', label='Frankfurt (Oder)', lat=52.34, lon=14.55, n_in_period=119),
        dict(id='pl:e:x:2', name='Frankfurt am Main', label='Frankfurt am Main', lat=50.112, lon=8.681, n_in_period=555),
        dict(id='pl:Q8', name='Newport', label='Newport', lat=50.70, lon=-1.29, n_in_period=18),
        dict(id='pl:Q9', name='Newport', label='Newport', lat=52.77, lon=-2.38, n_in_period=1),
        dict(id='pl:none', name='Nowhere', label='Nowhere', lat=None, lon=None, n_in_period=5),
        dict(id='pl:zero', name='Zero', label='Zero', lat=1.0, lon=1.0, n_in_period=0),
    ])
    g = B.merge_points(pg)
    assert g['pl:Q727' == pg['id']].iloc[0] == 'pl:e:ckcc:1' and g[pg['id'] == 'pl:e:ckcc:1'].iloc[0] == 'pl:e:ckcc:1'
    gid = dict(zip(pg['id'], g))
    assert gid['pl:Q1794'] == 'pl:e:x:2'                   # the Wikidata German label 'Frankfurt am Main' equals the register name
    assert gid['pl:Q4024'] == 'pl:Q4024'                   # the town on the Oder is 500 km away
    assert gid['pl:Q8'] == 'pl:Q8' and gid['pl:Q9'] == 'pl:Q9'      # two Newports far apart
    assert pd.isna(gid['pl:none']) and pd.isna(gid['pl:zero'])


def test_namecheck_keys_and_distance():
    assert N.nrm('Straßburg (Strasbourg)') == 'strassburg strasbourg'
    ks = N.name_keys('Halle (Saale), Stadt', ['Hallein'])
    assert 'halle' in ks and 'saale' in ks and 'stadt' not in ks and 'hallein' in ks
    assert abs(N.min_km(49.0, 8.0, np.array([[49.0, 8.0], [50.0, 8.0]]))) < 0.01
    assert 100 < N.min_km(49.0, 8.0, np.array([[50.0, 8.0]])) < 112


def test_overrides_file_is_well_formed():
    from pathlib import Path
    f = Path(__file__).resolve().parents[1] / 'data'
    o = B.load_overrides(f)
    assert len(o) >= 13 and o.loc[~o['ov_drop'], 'ov_lat'].between(-90, 90).all() and o['ov_drop'].sum() == 3      # v2.2: three register points dropped (Gärnitz, Longueville, Neuburg)
    sp = o[o['entity_key'] == 'ckcc_epistolarium:speyer.deu'].iloc[0]
    assert abs(sp['ov_lat'] - 49.32) < 0.01 and abs(sp['ov_lon'] - 8.43) < 0.01


def test_non_place_classes():
    assert B.is_non_place('Q783794:company|Q4830453:business')
    assert B.is_non_place('Q4167410:Wikimedia disambiguation page')
    assert B.is_non_place('Q16521:taxon')
    assert not B.is_non_place('Q515:city|Q1549591:big city') and not B.is_non_place('')


def test_merge_points_does_not_join_a_country_prefix_or_a_parenthesised_disambiguator():
    rows = [dict(id=f'pl:e:p:{i}', name=f'Portugal, Town{i}', label=f'Portugal, Town{i}', lat=38.7 + 0.01 * i, lon=-9.1,
                 n_in_period=1) for i in range(9)]                         # 9 different 'Portugal, X' within 20 km
    rows += [dict(id='pl:Q1', name='Anhalt, Fürstentum', label='Anhalt', lat=51.83, lon=12.25, n_in_period=1065),
             dict(id='pl:Q2', name='Coswig (Anhalt)', label='Coswig (Anhalt)', lat=51.88, lon=12.45, n_in_period=62),
             dict(id='pl:Q3', name='Danzig (Gdańsk)', label='Gdańsk', lat=54.35, lon=18.65, n_in_period=654),
             dict(id='pl:e:x', name='Danzig', label='Danzig', lat=54.35, lon=18.65, n_in_period=10)]
    g = dict(zip(*(lambda pg: (pg['id'], B.merge_points(pg)))(_pg(rows))))
    assert len({g[f'pl:e:p:{i}'] for i in range(9)}) == 9                  # all nine stay apart
    assert g['pl:Q1'] == 'pl:Q1' and g['pl:Q2'] == 'pl:Q2'
    assert g['pl:e:x'] == 'pl:Q3'                                          # 'Danzig' = 'Danzig (Gdańsk)' without its brackets
