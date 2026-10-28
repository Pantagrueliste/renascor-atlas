"""The fresh per-edition sample of the audit (src/ratlas/audit_editions.py): raw tag counts, header years, seeded draw, top-place flags."""
import numpy as np
import pandas as pd

from ratlas import audit_editions as E

TEI = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><author><persName>Anon</persName></author></titleStmt>
<sourceDesc><msDesc><history><origin><origDate when="1691-11-19"/></origin></history></msDesc></sourceDesc></fileDesc>
<profileDesc><creation><date notBefore="1658-01-01" notAfter="1658-12-31"/></creation></profileDesc></teiHeader>
<text><body><p><persName>Luther</persName> in <placeName>Wittenberg</placeName> and <rs type="person">er</rs><name type="person">Calvin</name></p></body></text></TEI>'''


def test_raw_profile_counts_tags_and_reads_every_kind_of_date():
    p = E.raw_profile(TEI, 'x/a.xml')
    assert p['fmt'] == 'xml' and p['tags']['persName'] == 2 and p['tags']['placeName'] == 1 and p['tags']['rs'] == 1 and p['tags']['name'] == 1
    assert p['n_core'] == 3                       # persName + placeName + orgName + geogName; rs and name are reported apart
    assert p['years'] == [1658, 1691]            # origDate and creation/date: the ssrq letter of 1691 is not a 1658 letter
    assert E.raw_profile(b'plain text', 'x/a.txt')['fmt'] == 'not-xml'
    assert E.raw_profile(b'<<<', 'x/a.xml')['fmt'] in ('unparsable', 'xml')


def test_the_draw_is_seeded_and_not_the_audits():
    m = pd.DataFrame({'coll_slug': ['a'] * 20 + ['b'] * 2, 'source_path': [f'f{i}.xml' for i in range(20)] + ['g1.xml', 'g2.xml']})
    s1 = E.sample_files(m, 'a', 'fresh-20261017')
    assert s1 == E.sample_files(m, 'a', 'fresh-20261017') and len(s1) == 3 and len(set(s1)) == 3
    assert s1 != E.sample_files(m, 'a', 'another-seed')
    assert sorted(E.sample_files(m, 'b', 'x')) == ['g1.xml', 'g2.xml']          # fewer than 3 files: all of them


def test_top_places_flags():
    pl = pd.DataFrame([
        dict(coll_slug='c', entity_key='k1', name='Roma', n_mentions=50, scope='edition', qid='Q220', qid_suggested=None, wd_label='Roma', lat=41.9,
             lon=12.5, coord_source='wikidata', coord_kind='point', place_kind='city', gc_conf=None, gc_evidence=None, note_type=None),
        dict(coll_slug='c', entity_key='k2', name='Spagna', n_mentions=40, scope='edition', qid='Q193190', qid_suggested=None, wd_label='Famagusta',
             lat=35.1, lon=33.9, coord_source='wikidata', coord_kind='point', place_kind='city', gc_conf=None, gc_evidence=None, note_type=None),
        dict(coll_slug='c', entity_key='k3', name='Nowhere', n_mentions=30, scope='edition', qid=None, qid_suggested='Q9', wd_label=None, lat=np.nan,
             lon=np.nan, coord_source='none', coord_kind=None, place_kind=None, gc_conf=None, gc_evidence=None, note_type=None),
        dict(coll_slug='c', entity_key='k4', name='Italia', n_mentions=20, scope='edition', qid=None, qid_suggested=None, wd_label=None, lat=42.8,
             lon=12.8, coord_source='geonames', coord_kind='area', place_kind='area', gc_conf=None, gc_evidence=None, note_type='country'),
        dict(coll_slug='c', entity_key='k5', name='Hidden', n_mentions=10, scope='excluded', qid=None, qid_suggested=None, wd_label=None, lat=1.0,
             lon=1.0, coord_source='register', coord_kind='point', place_kind=None, gc_conf=None, gc_evidence=None, note_type=None)])
    dist = pd.DataFrame({'coll_slug': ['c', 'c'], 'entity_key': ['k1', 'k4'], 'd_km': [0.5, 120.0], 'nearest_cc': ['IT', 'IT']})
    t = E.top_places(pl, dist, 'c').set_index('name')
    assert list(t.index) == ['Roma', 'Spagna', 'Nowhere', 'Italia']          # the excluded place is not listed
    assert t.loc['Roma', 'flags'] == ''
    assert 'name_vs_label' in t.loc['Spagna', 'flags']
    assert 'unplaced' in t.loc['Nowhere', 'flags'] and 'suggested_only' in t.loc['Nowhere', 'flags']
    assert t.loc['Italia', 'flags'] == 'area_point'                           # an area is not judged against its namesakes
