"""The nine checks of the audit of 2026-10-16 as blocking tests of the v2 tables (src/ratlas/audit_v2.py).

Two layers. (1) Always: the logic of every check on tiny synthetic tables, each with a case that must fail and a case that must pass, so a
change of a rule cannot silently weaken a check. (2) ``RATLAS_AUDIT_V2=1`` (the Slurm job ``slurm/audit_v2.sbatch`` sets it): every check on the
real v2 tables, zero failures required (reviewed exceptions live in ``data/audit_v2_allow.tsv`` with their reason).
"""
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ratlas import audit_compat as AC
from ratlas import audit_namecmp as AN
from ratlas import audit_v2 as A

# ---------------------------------------------------------------------------------------------------------------------
# the frozen yardsticks
# ---------------------------------------------------------------------------------------------------------------------


def test_frozen_matchers_still_know_the_audit_cases():
    assert AC.relation(AC.name_keys('Spagna', []), AC.auth_keys(['Famagusta']))[0].startswith('none')      # Spain -> Famagusta
    assert AC.relation(AC.name_keys('Prag', []), AC.auth_keys(['Prague', 'Praha', 'Prag']))[0] == 'exact'
    assert AC.relation(AC.name_keys('Patavia', []), AC.auth_keys(['Passau']))[0].startswith('none')
    assert AN.compare(['Martin Luther'], ['Martin Luther King'], set())[0] == 'ok'
    assert AN.compare(['Kirsten'], ['Hans Meier'], set())[0] == 'mismatch'
    assert AN.compare(['Georg Hohenlohe'], [], set())[0] == 'no_label'


# ---------------------------------------------------------------------------------------------------------------------
# check 1
# ---------------------------------------------------------------------------------------------------------------------
def _pair(**kw):
    d = dict(coll_slug='c', kind='person', entity_key='c:p', name='Kirsten', scheme='gnd', id='1', n_mentions=3, qid='Q1', src='wd', level='ok',
             label='Hans Meier', second_id='', status='accepted', reason='ok')
    d.update(kw)
    return d


def test_check1_person_mismatch_fails_unless_a_second_id_agrees():
    pairs = pd.DataFrame([_pair(), _pair(entity_key='c:q', name='Martin Luther', label='Martin Luther King', id='2'),
                          _pair(entity_key='c:r', name='Kirsten', second_id='viaf', id='3')])
    labels = {'c:p': ['Hans Meier'], 'c:q': ['Martin Luther King'], 'c:r': ['Hans Meier']}
    names = {('c', 'c:p'): ['Kirsten'], ('c', 'c:q'): ['Martin Luther'], ('c', 'c:r'): ['Kirsten']}
    pv = A.person_name_verdicts(pairs, lambda r: labels[r.entity_key], names, set())
    assert list(pv['verdict']) == ['mismatch', 'ok', 'mismatch']
    f = A.check_names(pairs, pv, pd.DataFrame(columns=pairs.columns.tolist() + ['verdict']))
    assert list(f['key']) == ['c:p']            # the one without a second id


def test_check1_place_none_fails_unless_second_id_or_coordinate_evidence():
    pairs = pd.DataFrame([
        _pair(kind='place', entity_key='c:spain', name='Spagna', label='Famagusta', src='wd', level='none0.00'),
        _pair(kind='place', entity_key='c:rome', name='Roma', label='Rome', src='wd', level='exact'),
        _pair(kind='place', entity_key='c:old', name='Patavia', label='Passau', src='wd', level='near_register'),
        _pair(kind='place', entity_key='c:two', name='Patavia', label='Passau', src='wd', level='none0.00', second_id='geonames')])
    labels = {'c:spain': ['Famagusta'], 'c:rome': ['Rome', 'Roma'], 'c:old': ['Passau'], 'c:two': ['Passau']}
    names = {('c', k): (n, []) for k, n in (('c:spain', 'Spagna'), ('c:rome', 'Roma'), ('c:old', 'Patavia'), ('c:two', 'Patavia'))}
    qv = A.place_name_verdicts(pairs, lambda r: labels[r.entity_key], names)
    f = A.check_names(pairs, pd.DataFrame(columns=pairs.columns.tolist() + ['verdict']), qv)
    assert list(f['key']) == ['c:spain']


def test_check1_a_confirmed_error_does_not_come_back():
    pairs = pd.DataFrame([_pair(kind='place', entity_key='cavriana:place-spagna', name='Spagna', scheme='tgn', id='7002327', qid='Q193190'),
                          _pair(kind='place', entity_key='cavriana:place-roma', name='Roma', scheme='tgn', id='7000874', qid='Q220')])
    conf = pd.DataFrame([dict(kind='place', coll_slug='c', entity_key='cavriana:place-spagna', entry_name='Spain', scheme='tgn', id='7002327',
                              qid='', bad_qid='', audit_check='place MISMATCH'),
                         dict(kind='place', coll_slug='c', entity_key='cavriana:other-key', entry_name='Spagna', scheme='tgn', id='7002327',
                              qid='', bad_qid='', audit_check='place MISMATCH'),
                         dict(kind='person', coll_slug='c', entity_key='c:p', entry_name='Davenant', scheme='wikidata', id='Q9', qid='Q9',
                              bad_qid='Q9', audit_check='nonhuman_qid_error')])
    loc = pd.DataFrame([dict(coll_slug='c', entity_key='c:p', qid='Q9', n_mentions=4), dict(coll_slug='c', entity_key='c:z', qid='Q9', n_mentions=0)])
    back = A.confirmed_wrong_back(pairs, conf, pd.DataFrame(), loc)
    assert set(back['kind']) == {'place', 'person'} and len(back) == 3   # same key; same folded name; the person still has the QID with mentions
    pairs2 = pairs[pairs['entity_key'] != 'cavriana:place-spagna']
    assert A.confirmed_wrong_back(pairs2, conf.iloc[:2], pd.DataFrame(), loc.iloc[:0]).empty
    # the key changed after the re-harvest but the folded name and the id are the same: still caught
    pairs3 = pd.DataFrame([_pair(kind='place', entity_key='c:new-key', name='SPAGNA', scheme='tgn', id='7002327')])
    assert len(A.confirmed_wrong_back(pairs3, conf.iloc[[0]].assign(entry_name='Spagna'), pd.DataFrame(), pd.DataFrame())) == 1


# ---------------------------------------------------------------------------------------------------------------------
# check 2
# ---------------------------------------------------------------------------------------------------------------------
def test_check2_non_human_qids():
    loc = pd.DataFrame([
        dict(coll_slug='c', entity_key='a', name='Zeus', qid='Q1', person_class='mythological'),
        dict(coll_slug='c', entity_key='b', name='Davenant', qid='Q2', person_class='human'),
        dict(coll_slug='c', entity_key='c', name='Kirsten', qid='Q3', person_class='human'),
        dict(coll_slug='c', entity_key='d', name='Cassel', qid='Q4', person_class='metonym'),
        dict(coll_slug='c', entity_key='e', name='Luther', qid='Q5x', person_class='human'),
        dict(coll_slug='c', entity_key='f', name='Hannover', qid='Q6', person_class='dynasty'),
        dict(coll_slug='c', entity_key='g', name='Nobody', qid=None, person_class='human')])
    items = {'Q1': ['Q178885'], 'Q2': ['Q4830453'], 'Q3': ['Q4167410'], 'Q4': ['Q5'], 'Q5x': ['Q5'], 'Q6': ['Q164950']}
    labels = {'Q178885': 'deity', 'Q4830453': 'business', 'Q4167410': 'Wikimedia disambiguation page', 'Q5': 'human', 'Q164950': 'house'}
    f = A.nonhuman_failures(loc, items, labels)
    assert set(f['key']) == {'b', 'c', 'd'}
    assert f.set_index('key').loc['c', 'why'].startswith('bad class')
    assert f.set_index('key').loc['d', 'why'] == 'metonym with an id'


# ---------------------------------------------------------------------------------------------------------------------
# check 3
# ---------------------------------------------------------------------------------------------------------------------
def test_check3_born_after_the_documents():
    pg = pd.DataFrame([
        dict(id='pe:Q1', qid='Q1', name='Nieto', label='Nieto', birth=1730.0, birth_precision=9, person_class_final='human'),
        dict(id='pe:Q2', qid='Q2', name='Near', label='Near', birth=1603.0, birth_precision=9, person_class_final='human'),     # within slack
        dict(id='pe:Q3', qid='Q3', name='Century', label='Century', birth=1700.0, birth_precision=7, person_class_final='human'),  # 17th century
        dict(id='pe:Q4', qid='Q4', name='Later', label='Later', birth=1650.0, birth_precision=9, person_class_final='human'),
        dict(id='pe:Q5', qid=None, name='NoQid', label=None, birth=np.nan, birth_precision=9, person_class_final='human')])
    ymax = pd.Series({'pe:Q1': 1700.0, 'pe:Q2': 1600.0, 'pe:Q3': 1610.0, 'pe:Q4': 1700.0})
    f = A.born_after_failures(pg, ymax)
    assert list(f['key']) == ['pe:Q1']


def test_check3_latest_document_uses_default_mentions_and_the_range_end():
    m = pd.DataFrame([
        dict(coll_slug='c', kind='person', local_key='c:p', doc_year=1600.0, doc_year_to=1600.0, excluded=False, canonical=True),
        dict(coll_slug='c', kind='person', local_key='c:p', doc_year=np.nan, doc_year_to=1650.0, excluded=False, canonical=True),
        dict(coll_slug='c', kind='person', local_key='c:p', doc_year=1900.0, doc_year_to=1900.0, excluded=True, canonical=True)])   # apparatus
    loc = pd.DataFrame([dict(coll_slug='c', entity_key='c:p', global_id='pe:Q1')])
    assert A.latest_document_year(m, loc)['pe:Q1'] == 1650.0


# ---------------------------------------------------------------------------------------------------------------------
# check 4
# ---------------------------------------------------------------------------------------------------------------------
def _fake_geonames(path: Path):
    rows = [  # id, name, ascii, alts, lat, lon, class, code, cc
        ('1', 'Kalkar', 'Kalkar', 'Calcar', '51.74', '6.29', 'P', 'PPL', 'DE'),
        ('2', 'Kalkar', 'Kalkar', '', '50.40', '6.60', 'P', 'PPL', 'DE'),            # the Eifel hamlet
        ('3', 'Mont Blanc', 'Mont Blanc', '', '45.83', '6.86', 'T', 'MT', 'FR'),
        ('4', 'Somewhere', 'Somewhere', '', '10', '10', 'R', 'RD', 'XX')]             # a road: not a place class
    txt = '\n'.join('\t'.join([a, b, c, d, e, f, g, h, i, '', '', '', '', '', '', '', '', '', '']) for a, b, c, d, e, f, g, h, i in rows)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('allCountries.txt', txt)


def _geo(**kw):
    d = dict(coll_slug='c', entity_key='k', name='Kalkar', lat=48.0, lon=11.0, coord_source='name_geonames', coord_kind='point', auth_lat=np.nan,
             auth_lon=np.nan, auth_name_level=None, n_mentions=3, qid=None, gc_cand=None)
    d.update(kw)
    return d


def test_check4_geonames_distance_and_failures(tmp_path):
    z = tmp_path / 'allCountries.zip'
    _fake_geonames(z)
    pl = pd.DataFrame({'coll_slug': ['c'] * 4, 'entity_key': ['k1', 'k2', 'k3', 'k4'], 'name': ['Kalkar', 'Kalkar', 'Atlantis', 'Calcar'],
                       'variants': [[], [], [], []], 'lat': [51.74, 50.40, 0.0, 51.7], 'lon': [6.29, 6.60, 0.0, 6.3]})
    d = A.geonames_distances(pl, z, log=lambda *_: None).set_index('entity_key')
    assert d.loc['k1', 'd_km'] < 1 and d.loc['k2', 'd_km'] < 1          # each is on one of the two Kalkars
    assert d.loc['k3', 'd_km'] == -1                                    # no feature of that name at all
    assert d.loc['k4', 'd_km'] < 10                                     # the alternate name
    assert d.loc['k1', 'n_settlement'] == 2 and d.loc['k1', 'n_polity'] == 0
    # the audit's phonetic fold: Calcar and Kalkar are one name (c/k), so a point on Kalkar is also near 'Calcar'
    geo = pd.DataFrame([_geo(entity_key='k1', name='Kalkar'),                                                         # in Bavaria: far from both
                        _geo(entity_key='k2', coord_kind='area', lat=50.4, lon=6.6),                                  # an area: not judged
                        _geo(entity_key='k3', name='Atlantis', lat=0.0, lon=0.0),                                     # no namesake: not judged
                        _geo(entity_key='k4', name='Calcar', coord_source='name_exonym', lat=0.0, lon=0.0)])           # curated table
    dist = A.geonames_distances(geo.assign(variants=[[]] * 4), z, log=lambda *_: None)
    fail, review = A.far_from_every_namesake(geo, dist)
    assert list(fail['key']) == ['k1'] and review.empty


def test_check4_who_vouches_for_a_point(tmp_path):
    z = tmp_path / 'allCountries.zip'
    _fake_geonames(z)
    geo = pd.DataFrame([
        _geo(entity_key='a', coord_source='wikidata', auth_name_level='exact', auth_lat=48.0, auth_lon=11.0),         # an accepted authority's point: vouched
        _geo(entity_key='b', coord_source='wikidata', auth_name_level='token', auth_lat=48.0, auth_lon=11.0),         # (its identity is check 1's business)
        _geo(entity_key='c', coord_source='register', auth_name_level='exact', auth_lat=48.1, auth_lon=11.1),         # register + authority within 50 km
        _geo(entity_key='d', coord_source='register'),                                                                 # editors' point alone: review
        _geo(entity_key='e', coord_source='name_propagation', gc_cand='c:d'),                                         # a copy of that point: review
        _geo(entity_key='f', coord_source='name_propagation', gc_cand='c:b'),                                         # a copy of an authority point: vouched
        _geo(entity_key='g', coord_source='name_geonames'),                                                            # our own inference: fails
        _geo(entity_key='h', coord_source='name_propagation', gc_cand='c:g'),                                         # a copy of that: fails
        _geo(entity_key='i', coord_source='name_propagation', gc_cand='c:h'),                                         # a copy of the copy: fails
        _geo(entity_key='j', coord_source='name_propagation', gc_cand='zz:nowhere')])                                  # a copy of nothing we know: fails
    dist = A.geonames_distances(geo.assign(variants=[[]] * len(geo)), z, log=lambda *_: None)
    fail, review = A.far_from_every_namesake(geo, dist)
    assert set(fail['key']) == {'g', 'h', 'i', 'j'} and set(review['key']) == {'d', 'e'}


def test_check4_qualifier_words_are_not_part_of_the_name():
    assert A.fkey('wing') in A.place_keys('Wing. parish', [])
    assert A.fkey('schildesche') in A.place_keys('Stift Schildesche', [])
    assert A.fkey('middlesex') in A.place_keys('County of Middlesex', [])
    assert A.fkey('kalkar') in A.place_keys('Kalkar, Stadt', [])


def test_check4_shared_feature_evidence(tmp_path):
    z = tmp_path / 'g.zip'
    rows = [('1', 'Venezia', 'Venezia', 'Venedig,Venice', '45.44', '12.33', 'P', 'PPLA', 'IT'),
            ('2', 'Altstadt', 'Altstadt', '', '51.48', '11.97', 'P', 'PPLX', 'DE'),
            ('3', 'Halle', 'Halle', 'Halle Saale', '51.4825', '11.97', 'P', 'PPLA3', 'DE')]
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr('allCountries.txt', '\n'.join('\t'.join(list(r[:6]) + [r[6], r[7], r[8]] + [''] * 10) for r in rows))
    items = pd.DataFrame({'entry_keys': [A.place_keys('Venedig', []), A.place_keys('Halle', [])],
                          'label_keys': [{A.fkey(k) for k in AC.auth_keys(['Venezia'])}, {A.fkey(k) for k in AC.auth_keys(['Altstadt'])}],
                          'lat': [45.44, 51.48], 'lon': [12.33, 11.97]})
    ok = A.shared_feature_evidence(items, z, log=lambda *_: None)
    assert list(ok) == [True, False]       # Venedig and Venezia are names of one feature; Halle and Altstadt are two features


# ---------------------------------------------------------------------------------------------------------------------
# check 5
# ---------------------------------------------------------------------------------------------------------------------
def test_check5_capital_for_country():
    base = dict(coll_slug='c', qid=None, note_type=None, gn_fcode=None, coord_kind='point', coord_source='register', place_kind='city', n_mentions=5)
    pl = pd.DataFrame([
        dict(base, entity_key='it', name='Italia', note_type='country'),                                           # the capital's register point
        dict(base, entity_key='by', name='Bayern', qid='Q980', gn_fcode='ADM1'),                                    # state, kind wrongly 'city'
        dict(base, entity_key='is', name='Konstantinopel', qid='Q406', gn_fcode='ADM1'),                            # a city that is also a province
        dict(base, entity_key='ar', name='Italia', note_type='country', coord_kind='area', coord_source='geonames'),   # fine: an area
        dict(base, entity_key='fr', name='Frankreich', qid='Q142', gn_fcode='PCLI', place_kind='area'),
        dict(base, entity_key='ro', name='Roma', qid='Q220', gn_fcode='PPLC')])
    cls = {'Q980': ['federated state of Germany', 'free state', 'ward'], 'Q406': ['big city', 'city'], 'Q142': ['sovereign state', 'country'],
           'Q220': ['capital', 'city']}
    f = A.capital_for_country(pl, lambda q: cls.get(q, []))
    assert set(f['key']) == {'it', 'by', 'fr'}


def test_check5_a_name_only_countries_carry():
    row = dict(coll_slug='c', qid=None, note_type=None, gn_fcode=None, coord_kind='point', place_kind=None, n_mentions=9)
    pl = pd.DataFrame([dict(row, entity_key='pol', name='Polen, Königreich', coord_source='name_propagation'),
                       dict(row, entity_key='sic', name='Sicilien', coord_source='name_propagation'),
                       dict(row, entity_key='lux', name='Luxemburg', coord_source='name_geonames'),             # country AND city: not a polity name
                       dict(row, entity_key='ita', name='Italia', coord_source='name_propagation', coord_kind='area'),
                       dict(row, entity_key='vat', name='Vaticana', coord_source='name_propagation'),
                       dict(row, entity_key='par', name='Parish of St. Andrew (Holborn)', coord_source='register'),
                       dict(row, entity_key='zee', name='Zelanda', coord_source='name_exonym')])
    dist = pd.DataFrame({'coll_slug': ['c'] * 7, 'entity_key': ['pol', 'sic', 'lux', 'ita', 'vat', 'par', 'zee'],
                         'n_polity': [1] * 7, 'n_settlement': [0, 0, 4, 0, 0, 0, 0],
                         'd_seat_km': [1.2, 0.7, 0.5, 20.0, 0.5, 2.0, 12.9], 'd_polity_km': [216.0, 0.7, 0.5, 5.0, 0.4, 5000.0, 3.0],
                         'seat_name': ['Krakow', 'Enna', 'Luxembourg', 'Roma', 'Vatican City', 'London', 'Middelburg'],
                         'polity_hit': ['x'] * 7})
    f, review = A.polity_name_points(pl, dist)
    # Krakow for the Kingdom of Poland: on a big seat, 216 km from the polity's own point. Sicily's centroid happens to be on Enna and Vatican
    # City is a polity AND its seat: the polity's own point. A London parish folds onto the parishes of Barbados: not a polity.
    assert list(f['key']) == ['pol'] and set(review['key']) == {'sic', 'vat', 'zee'}


def test_check5_geonames_scan_finds_the_seat_and_the_polity(tmp_path):
    z = tmp_path / 'g.zip'
    rows = [('1', 'Polska', 'Polska', 'Polen', '52.0', '20.0', 'A', 'PCLI', 'PL'),
            ('2', 'Krakow', 'Krakow', 'Krakau', '50.05', '19.93', 'P', 'PPLA', 'PL')]
    with zipfile.ZipFile(z, 'w') as zf:
        # a GeoNames line has 19 fields; population is the 15th (index 14)
        zf.writestr('allCountries.txt', '\n'.join('\t'.join(list(r[:6]) + [r[6], r[7], r[8]] + ['', '', '', '', '', '760000', '', '', '', '']) for r in rows))
    pl = pd.DataFrame({'coll_slug': ['c'], 'entity_key': ['pol'], 'name': ['Polen, Königreich'], 'variants': [[]], 'lat': [50.05], 'lon': [19.93]})
    d = A.geonames_distances(pl, z, log=lambda *_: None).iloc[0]
    assert d['n_polity'] == 1 and d['n_settlement'] == 0 and d['d_seat_km'] < 1 and d['seat_name'] == 'Krakow'
    assert 200 < d['d_polity_km'] < 230          # Poland's own point is the centroid, 216 km from Krakow


# ---------------------------------------------------------------------------------------------------------------------
# check 6
# ---------------------------------------------------------------------------------------------------------------------
def _m(**kw):
    d = dict(coll_slug='c', source_path='f.xml', doc_id='d', doc_year=1600.0, doc_year_from=1600.0, doc_year_to=1600.0, date_source='header',
             in_period=True, kind='person', tag='persName', role='mention', zone='text', person_class='human', excluded=False, excluded_reason=None,
             surface='Martin Luther', local_key='c:luther', canonical=True)
    d.update(kw)
    return d


def test_check6_default_mentions():
    m = pd.DataFrame([
        _m(),
        _m(zone='editorial_commentary', surface='Calvin'),
        _m(zone='footnote', surface='Zwingli'),
        _m(person_class='character', surface='Hamlet'),
        _m(kind='place', person_class='celestial', surface='Mars', local_key=None),
        _m(surface='S.l.', kind='place', local_key=None, person_class=None),
        _m(surface='ich', local_key=None),
        _m(role='provenance', surface='Oxford', kind='place'),
        _m(zone='apparatus', surface='X', excluded=True),         # excluded: fine
        _m(surface='Er', local_key=None, excluded=True)])         # excluded: fine
    f = A.default_mention_failures(m)
    assert set(f['why']) == {'zone', 'class', 'role', 'placeholder'} and len(f) == 6 and 'Martin Luther' not in set(f['surface'])
    # pronouns tagged by the editors are a policy question: reported, failing only on request
    assert set(A.default_mention_failures(m, with_pronouns=True)['why']) == {'zone', 'class', 'role', 'placeholder', 'pronoun'}
    assert A.pronoun_mentions(m).set_index('coll_slug').loc['c', 'n_mentions'] == 1


def test_check6_tables_hold_the_default_counts():
    m = pd.DataFrame([_m(), _m(surface='Luther again'), _m(excluded=True)])
    ok = pd.DataFrame({'n_mentions': [2]})
    assert A.count_failures(m, ok, 'person') == []
    bad = A.count_failures(m, pd.DataFrame({'n_mentions': [3]}), 'person')
    assert bad and bad[0]['n_mentions'] == 1


# ---------------------------------------------------------------------------------------------------------------------
# check 7
# ---------------------------------------------------------------------------------------------------------------------
def test_check7_duplicates():
    cb = [_m(coll_slug='casebooks', doc_id='CASE1', zone='header', local_key='casebooks:P1', tag='person@particDesc') for _ in range(2)]
    fine = [_m(coll_slug='casebooks', doc_id='CASE2', zone='header', local_key='casebooks:P1', tag='person@particDesc'),
            _m(coll_slug='casebooks', doc_id='CASE3', zone='header', local_key='casebooks:P1', tag='person@particDesc')]
    f, info = A.duplicate_failures(pd.DataFrame(cb + fine))
    assert set(f['coll_slug']) == {'casebooks'} and f['n_mentions'].sum() >= 1
    f2, _ = A.duplicate_failures(pd.DataFrame(fine))
    assert f2.empty
    # an edition that repeats header rows beyond 3 %
    rows = [_m(coll_slug='x', doc_id='d', zone='header', local_key=f'x:{i}', tag='meta') for i in range(20)] + \
           [_m(coll_slug='x', doc_id='d', zone='header', local_key='x:0', tag='meta')] * 3
    f3, info3 = A.duplicate_failures(pd.DataFrame(rows), strict_header=True)
    assert list(f3['coll_slug']) == ['x'] and info3['x'] > 0.03
    f4, _ = A.duplicate_failures(pd.DataFrame(rows))              # a multi-letter file repeats its senders legitimately: information only
    assert f4.empty


# ---------------------------------------------------------------------------------------------------------------------
# check 8
# ---------------------------------------------------------------------------------------------------------------------
def test_check8_dates():
    m = pd.DataFrame([
        _m(),                                                                                                  # ok
        _m(doc_year=1596.0, doc_year_from=1596.0, doc_year_to=1634.0, date_source='collection'),                # midpoint of a wide range
        _m(doc_year=np.nan, doc_year_from=1596.0, doc_year_to=1634.0, date_source='collection'),                 # ok: no doc_year
        _m(doc_year=1620.0, doc_year_from=1600.0, doc_year_to=1610.0),                                           # outside its range
        _m(doc_year=1800.0, doc_year_from=1800.0, doc_year_to=1800.0, in_period=True),                           # outside the period, in_period True
        _m(doc_year=np.nan, doc_year_from=1650.0, doc_year_to=1750.0, in_period=True, date_source='collection')])  # straddles: must be None
    f = A.date_failures(m)
    assert set(f['why']) == {'a range wider than 20 years has a doc_year (midpoint)', 'doc_year outside its own range',
                             'range outside the period but in_period is not False', 'wide range straddling the period boundary has an in_period'}


# ---------------------------------------------------------------------------------------------------------------------
# check 9
# ---------------------------------------------------------------------------------------------------------------------
def test_check9_sources(tmp_path):
    assert A.source_prefix('originals_2026-10-17/abacus/abacus.zip!x/y.xml') == 'originals_2026-10-17/abacus'
    assert A.source_prefix('eng_dh/casebooks-data-master/cases/CASE1.xml') == 'eng_dh'
    pf = pd.DataFrame([dict(coll_slug='new', prefix='originals_2026-10-17/new'), dict(coll_slug='old', prefix='eng_dh'),
                       dict(coll_slug='lost', prefix='cidtc'), dict(coll_slug='stale', prefix='eng_dh')])
    legacy = pd.DataFrame([dict(slug='old', source_prefix='eng_dh'), dict(slug='stale', source_prefix='eng_dh'),
                           dict(slug='gone', source_prefix='zzz')])
    fresh = pd.DataFrame([dict(slug='stale', changed_since_harvest='yes'), dict(slug='new', changed_since_harvest='yes'),
                          dict(slug='old', changed_since_harvest='no')])
    f = A.sources_failures(pf, lambda p: p.startswith('originals_'), legacy, fresh)
    why = {(r.coll_slug, r.why.split(' ')[0]) for r in f.itertuples(index=False)}
    assert ('lost', 'no') in why                       # neither SOURCES.tsv nor legacy
    assert ('gone', 'legacy') in why                   # a legacy row nobody reads
    assert ('stale', 'changed') in why                 # changed upstream, still legacy
    assert not any(r.coll_slug in ('old', 'new') for r in f.itertuples(index=False))


def test_check9_manifest(tmp_path):
    import hashlib
    d = tmp_path / 'f'
    d.mkdir()
    (d / 'a.xml').write_bytes(b'<a/>')
    (d / 'b.xml').write_bytes(b'<bb/>')
    h = hashlib.sha256(b'<a/>').hexdigest()
    pd.DataFrame({'file': ['a.xml', 'b.xml', 'gone.xml'], 'bytes': ['4', '9', '1'], 'sha256': [h, 'x', 'y']}).to_csv(d / 'MANIFEST.tsv', sep='\t', index=False)
    rows = A.manifest_failures(d)
    assert {r['why'].split(' ')[0] for r in rows} == {'size', 'listed'}
    assert A.manifest_failures(d, verify_hashes=True) == rows


# ---------------------------------------------------------------------------------------------------------------------
# allow list
# ---------------------------------------------------------------------------------------------------------------------
def test_allow_list_moves_reviewed_failures_out():
    r = A.Result(4, 'x', 'x', failures=pd.DataFrame({'coll_slug': ['c', 'c'], 'key': ['k1', 'k2']}))
    allow = pd.DataFrame([dict(check='4', coll_slug='c', key='k1', reason='reviewed')])
    A.apply_allow(r, allow)
    assert list(r.failures['key']) == ['k2'] and r.allowed == 1
    r2 = A.Result(5, 'x', 'x', failures=pd.DataFrame({'coll_slug': ['c'], 'key': ['k1']}))
    A.apply_allow(r2, allow)
    assert len(r2.failures) == 1                 # other check: untouched


def test_allow_list_rows_are_documented():
    t = A.load_allow()
    if len(t):
        assert (t['reason'].str.len() > 15).all(), 'every exception carries its reason'
        assert set(t['check']) <= set(map(str, range(1, 10)))


# ---------------------------------------------------------------------------------------------------------------------
# the loaders against the real build of the synthetic edition (schema drift guard)
# ---------------------------------------------------------------------------------------------------------------------
def test_checks_run_on_a_built_synthetic_v2(tmp_path):
    from ratlas import reconcile_v2 as V
    from test_reconcile_v2 import _make
    v2, rd = _make(tmp_path)
    out = tmp_path / 'out'
    V.build(v2, rd, out)
    T = A.Tables(v2, out, rd, None, tmp_path / 'audit')
    for n in (2, 3, 5, 6, 7, 8):
        r = A.CHECKS[n](T)
        assert isinstance(r, A.Result) and r.skipped == ''
    # the synthetic build must satisfy the rules that its own gates implement
    assert A.run_check8(T).ok and A.run_check7(T).ok and A.run_check3(T).ok
    r6 = A.run_check6(T)
    assert r6.ok, r6.failures.to_string()
    r2 = A.run_check2(T)
    assert r2.ok, r2.failures.to_string()
    r1 = A.run_check1(T)
    assert r1.ok, r1.failures.to_string()
    A.write_results([r1, r2, r6], tmp_path / 'audit')
    assert (tmp_path / 'audit' / 'audit_v2_summary.json').exists()


def test_check1_reads_the_accepted_tgn_ids_in_the_local_index(tmp_path):
    """v2.1: a TGN id accepted by the gates is judged by the audit's frozen matcher on the subject's names in the local Getty index."""
    from ratlas import reconcile_v2 as V
    from test_reconcile_gates import _tgn_dir
    from test_reconcile_v2 import _entity, _make, _mention
    rows = [_mention('colla', 'place', 'Ratisbona', key='colla:rat', reg=True, auth={'tgn': '7000874'})]
    ents = [_entity('colla', 'place', 'colla:rat', 'Ratisbona', auth={'tgn': '7000874'})]
    v2, rd = _make(tmp_path, rows, ents)
    out = tmp_path / 'out'
    tg = _tgn_dir(tmp_path)
    V.build(v2, rd, out, tgn_dir=tg)
    assert A.run_check1(A.Tables(v2, out, rd, None, tmp_path / 'audit', tg)).ok
    pairs = pd.read_parquet(out / 'gate_pairs.parquet')
    assert ((pairs['scheme'] == 'tgn') & (pairs['status'] == 'accepted') & (pairs['src'] == 'tgn')).sum() == 1
    # the entry's names have nothing to do with the subject's names any more (as if the gate had accepted a wrong id): the audit's matcher sees it
    pl = pd.read_parquet(v2 / 'places_local.parquet')
    pl['variants'] = [['Zzyzx'] if k == 'colla:rat' else v for k, v in zip(pl['entity_key'], pl['variants'])]
    pl['name'] = ['Zzyzx' if k == 'colla:rat' else v for k, v in zip(pl['entity_key'], pl['name'])]
    pl.to_parquet(v2 / 'places_local.parquet', index=False)
    bad = A.run_check1(A.Tables(v2, out, rd, None, tmp_path / 'audit', tg))
    assert not bad.ok and set(bad.failures['scheme']) == {'tgn'}
    assert A.run_check1(A.Tables(v2, out, rd, None, tmp_path / 'audit2')).ok            # without the index the id cannot be judged


# ---------------------------------------------------------------------------------------------------------------------
# the real tables
# ---------------------------------------------------------------------------------------------------------------------
REAL = os.environ.get('RATLAS_AUDIT_V2') == '1'
real = pytest.mark.skipif(not REAL, reason='RATLAS_AUDIT_V2=1 (Slurm job slurm/audit_v2.sbatch) runs the checks on the real v2 tables')


@pytest.fixture(scope='module')
def tables():
    d = Path.home() / 'renascor-atlas-data'
    g = os.environ.get
    return A.Tables(g('RATLAS_AUDIT_ALL', d / 'out_v2' / 'all'), g('RATLAS_AUDIT_FINAL', d / 'out_v2' / 'reconcile' / 'final'),
                    g('RATLAS_AUDIT_RDIR', d / 'reconcile_v2'), g('RATLAS_AUDIT_RAW', d / 'raw'), g('RATLAS_AUDIT_OUT', d / 'out_v2' / 'audit_v2'),
                    g('RATLAS_AUDIT_TGN', d / 'tgn'))


@pytest.fixture(scope='module')
def allow():
    return A.load_allow()


def _assert_clean(res: A.Result):
    assert res.skipped == '', res.skipped
    assert res.n_checked > 0
    assert res.ok, f'check {res.n} {res.name}: {len(res.failures)} failures\n' + res.failures.head(30).to_string()


@real
@pytest.mark.parametrize('n', range(1, 10))
def test_real_v2_tables(tables, allow, n):
    zp = os.environ.get('RATLAS_AUDIT_GEONAMES')
    zp = Path(zp) if zp else None
    if n == 1:
        res = A.run_check1(tables, zp)
    elif n == 4:
        res = A.run_check4(tables, None)               # the distances were computed by the CLI run just before (gn_final_dist.tsv)
    elif n == 9:
        res = A.run_check9(tables, False)
    else:
        res = A.CHECKS[n](tables)
    _assert_clean(A.apply_allow(res, allow))


def test_check1_a_manual_review_of_a_person_stands():
    """After the audit a person read the sources: a modern scholar cited in the editor's notes is not a wrong id (reason manual_review)."""
    conf = pd.DataFrame([dict(kind='person', coll_slug='c', entity_key='c:bonnet', entry_name='Jules Bonnet', scheme='wikidata', id='Q3188439',
                              qid='Q3188439', bad_qid='Q3188439', audit_check='chronology_birth_after_documents'),
                         dict(kind='person', coll_slug='c', entity_key='c:other', entry_name='Davenant', scheme='wikidata', id='Q9', qid='Q9',
                              bad_qid='Q9', audit_check='nonhuman_qid_error')])
    pairs = pd.DataFrame([_pair(kind='person', entity_key='c:bonnet', qid='Q3188439', reason='manual_review'),
                          _pair(kind='person', entity_key='c:other', qid='Q9', reason='ok')])
    loc = pd.DataFrame([dict(coll_slug='c', entity_key='c:bonnet', qid='Q3188439', n_mentions=3), dict(coll_slug='c', entity_key='c:other', qid='Q9', n_mentions=3)])
    back = A.confirmed_wrong_back(pairs, conf, pd.DataFrame(), loc)
    assert list(back['key']) == ['c:other']


def test_check4_runs_from_the_cached_distances(tmp_path):
    """The pytest on the real tables does not rescan GeoNames: it reads gn_final_dist.tsv (with the coordinates it was computed for)."""
    final = tmp_path / 'final'
    final.mkdir()
    out = tmp_path / 'audit'
    out.mkdir()
    pl = pd.DataFrame([_geo(entity_key='a', name='Kalkar', lat=48.0, lon=11.0, scope='edition', coord_source='name_geonames'),
                       _geo(entity_key='b', name='Roma', lat=41.9, lon=12.5, scope='edition', coord_source='register')])
    pl.to_parquet(final / 'places_local_geo.parquet')
    pd.DataFrame({'coll_slug': ['c', 'c'], 'entity_key': ['a', 'b'], 'd_km': [120.0, 0.0], 'nearest_gid': ['1', '2'], 'nearest_cc': ['DE', 'IT'],
                  'lat': [48.0, 41.9], 'lon': [11.0, 12.5]}).to_csv(out / 'gn_final_dist.tsv', sep='\t', index=False)
    res = A.run_check4(A.Tables(tmp_path, final, None, None, out), None)
    assert res.skipped == '' and list(res.failures['key']) == ['a']
    pl.assign(lat=[49.0, 41.9]).to_parquet(final / 'places_local_geo.parquet')       # the tables moved on: the cache is stale, the check says so
    assert 'older than the tables' in A.run_check4(A.Tables(tmp_path, final, None, None, out), None).skipped


def test_check5_reads_the_head_of_a_class_label_not_the_country_in_it():
    """v2.2: 'borough in the United Kingdom' (Bath) and 'cadastral area in the Czech Republic' are not "Wikidata state / country class"."""
    from ratlas import audit_v2 as A
    pl = pd.DataFrame([dict(coll_slug='a', entity_key='bath', name='Bath', qid='Q1', note_type='place', gn_fcode='ADM2', place_kind='city', coord_kind='point', coord_source='register', n_mentions=1),
                       dict(coll_slug='a', entity_key='fr', name='France', qid='Q2', note_type='place', gn_fcode='', place_kind='city', coord_kind='point', coord_source='register', n_mentions=1),
                       dict(coll_slug='a', entity_key='sax', name='Sachsen', qid='Q3', note_type='place', gn_fcode='', place_kind='city', coord_kind='point', coord_source='register', n_mentions=1)])
    labs = {'Q1': ['unitary authority area', 'borough in the United Kingdom'], 'Q2': ['sovereign state', 'country'], 'Q3': ['state of Germany']}
    f = A.capital_for_country(pl, lambda q: labs[q])
    assert set(f['key']) == {'fr', 'sax'}
