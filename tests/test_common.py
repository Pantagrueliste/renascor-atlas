from collections import Counter

import pyarrow.parquet as pq
import pytest

from ratlas import common as C
from ratlas.model import Mention, Entity, MENTION_SCHEMA, ENTITY_SCHEMA

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'

LETTER = f'''<?xml version="1.0"?>
<TEI {NS} xml:id="L1"><teiHeader><fileDesc>
 <titleStmt><title>Letter</title><author><persName ref="#p1">Erasmus</persName></author>
   <respStmt><resp>edited by</resp><persName>Modern Editor</persName></respStmt></titleStmt>
 <publicationStmt><date when="2019-05-01"/></publicationStmt>
 <sourceDesc><p>x</p></sourceDesc></fileDesc>
 <profileDesc><correspDesc>
  <correspAction type="sent"><persName ref="#p1">Erasmus</persName><placeName ref="https://www.geonames.org/2988507/paris.html">Paris</placeName><date when="1525-03-02"/></correspAction>
  <correspAction type="received"><persName ref="d-nb.info/gnd/118540238 #p2">Goethe</persName><placeName key="#pl2">Basel</placeName></correspAction>
 </correspDesc></profileDesc></teiHeader>
 <standOff><listPerson><person xml:id="p1"><persName type="display">Erasmus, Desiderius</persName><persName>Erasmus Roterodamus</persName>
   <idno type="VIAF">95227332</idno><birth when="1466"/><death>1536</death></person>
  <person xml:id="p2"><persName><forename>Peter</forename><surname>Gast</surname></persName><idno>Q90</idno></person></listPerson>
  <listPlace><place xml:id="pl2"><placeName>Basel</placeName><location><geo>47.55 7.59</geo></location>
   <idno type="geonames">2661604</idno></place></listPlace></standOff>
 <text xml:lang="la"><body><opener><dateline><placeName>Lutetiae</placeName></dateline></opener>
 <p>Ad <persName ref="#p2">Petrum</persName> et <rs type="place" ref="#pl2">Basileam</rs>, <rs type="plac">Roma</rs>,
  <name>Foo</name>, <orgName>Senatus <placeName>Venetus</placeName></orgName>;
  <choice><abbr>Fr.</abbr><expan>Franciscus</expan></choice> <persName>X</persName>.</p>
 <closer><signed><persName>Erasmus</persName></signed></closer></body></text></TEI>'''


def rec(name='L1.xml'):
    return C.FileRecord(path='coll/' + name, coll='coll')


def ctx(tier='T1', **kw):
    return C.Context(slug='demo', licence='CC-BY-4.0', tier=tier, renascor_ids=['demo-entry'], **kw)


def test_classify_token():
    cases = {
        'https://d-nb.info/gnd/118540238': ('gnd', '118540238'),
        'http://d-nb.info/gnd/4001328-5': ('gnd', '4001328-5'),
        'https://d-nb.info/gnd/11856322X': ('gnd', '11856322X'),
        'gnd:118540238': ('gnd', '118540238'),
        '(DE-588)118540238': ('gnd', '118540238'),
        'https://viaf.org/viaf/95227332/': ('viaf', '95227332'),
        'http://www.wikidata.org/entity/Q90': ('wikidata', 'Q90'),
        'wd:Q90': ('wikidata', 'Q90'), 'Q90': ('wikidata', 'Q90'),
        'https://www.geonames.org/2988507/paris.html': ('geonames', '2988507'),
        'http://sws.geonames.org/2988507/': ('geonames', '2988507'),
        'http://vocab.getty.edu/tgn/7008038': ('tgn', '7008038'),
        'http://vocab.getty.edu/page/ulan/500115493': ('ulan', '500115493'),
        'https://pleiades.stoa.org/places/423025': ('pleiades', '423025'),
        'https://hls-dhs-dss.ch/de/articles/012345/2010-01-01/': ('hls', '012345'),
        'https://isni.org/isni/0000000121251078': ('other', 'isni:0000000121251078'),
        'https://de.wikipedia.org/wiki/Kleopatra_VII.': ('other', 'wikipedia:de:Kleopatra_VII.'),
        '#p1': ('local', 'p1'), 'register.xml#p1': ('local', 'p1'), 'psn:foo': ('local', 'psn:foo'),
        'p0001': ('local', 'p0001'), '#': ('none', ''), '': ('none', ''),
    }
    for tok, exp in cases.items():
        assert C.classify_token(tok) == exp, tok
    assert C.classify_token('https://example.org/x')[0] == 'other'


def test_parse_auth_and_merge():
    a = C.parse_auth(['https://d-nb.info/gnd/118540238 Q90', 'gnd:1234567-8'])
    assert a['gnd'] == '118540238' and a['wikidata'] == 'Q90' and 'gnd:1234567-8' in a['other']
    m = C.merge_auth(a, {'viaf': '1', 'gnd': '999999999'})
    assert m['viaf'] == '1' and m['gnd'] == '118540238'


def test_header_dates_letter_and_ignore_publication():
    root = C.parse_xml(LETTER.encode())
    d = C.header_dates(root)
    assert (d.year_from, d.year_to, d.source) == (1525, 1525, 'letter')   # not the 2019 publication date


def test_header_dates_variants():
    mk = lambda hdr, body='': C.parse_xml(f'<TEI {NS}><teiHeader><fileDesc>{hdr}</fileDesc></teiHeader><text><body>{body}</body></text></TEI>'.encode())
    d = C.header_dates(mk('<sourceDesc><bibl><date>Printed 1573-1575</date></bibl></sourceDesc>'))
    assert (d.year_from, d.year_to, d.source) == (1573, 1575, 'header')
    d = C.header_dates(mk('<publicationStmt><date>2020</date></publicationStmt>'))
    assert not d.known
    d = C.header_dates(mk('', '<opener><dateline><date notBefore="1601-01" notAfter="1602-12">anno 1601</date></dateline></opener>'))
    assert (d.year_from, d.year_to, d.source) == (1601, 1602, 'letter')
    r = C.resolve_doc_date(mk(''), C.FileRecord(path='a/letter_1555_02.xml', coll='a'), filename_dates=True)
    assert (r.year_from, r.source) == (1555, 'filename')
    r = C.resolve_doc_date(mk(''), None, default_years=(1500, 1599))
    assert (r.year_from, r.year_to, r.source) == (1500, 1599, 'collection')


def test_scan_mentions_roles_and_skips():
    root = C.parse_xml(LETTER.encode())
    hits = C.scan_mentions(root, want_context=True, counters=(cnt := Counter()))
    got = [(h.zone, h.tag, h.role, h.surface) for h in hits]
    assert ('hdr', 'persName', 'author', 'Erasmus') in got
    assert ('hdr', 'persName', 'sender', 'Erasmus') in got
    assert ('hdr', 'placeName', 'sent_from', 'Paris') in got
    assert ('hdr', 'persName', 'addressee', 'Goethe') in got
    assert ('hdr', 'placeName', 'sent_to', 'Basel') in got
    assert ('txt', 'placeName', 'sent_from', 'Lutetiae') in got            # dateline
    assert ('txt', 'persName', 'signer', 'Erasmus') in got
    assert ('txt', 'rs@place', 'mention', 'Basileam') in got
    assert ('txt', 'rs@plac', 'mention', 'Roma') in got                      # typo type mapped
    assert ('txt', 'orgName', 'mention', 'Senatus Venetus') in got
    assert ('txt', 'placeName', 'mention', 'Venetus') in got                 # nested other-kind kept
    assert not any(s == 'Modern Editor' for *_, s in got)                   # respStmt skipped
    assert not any(s.startswith('Erasmus, ') or s == 'Peter Gast' for *_, s in got)   # register skipped
    assert cnt['untyped_rs_name'] == 1                                       # <name>Foo</name>
    assert any(h.surface == 'X' and 'Franciscus' not in h.context for h in hits)
    assert all(h.lang == 'la' for h in hits if h.zone == 'txt')
    assert all(len(h.context) <= 120 for h in hits)
    ctxs = {h.surface: h.context for h in hits if h.zone == 'txt'}
    assert 'Fr.' in ctxs['X'] and 'Franciscus' not in ctxs['X']             # diplomatic reading in context


def test_register_parsing():
    root = C.parse_xml(LETTER.encode())
    ents, noid = C.parse_register(root, ctx(), 'p.xml')
    by = {e.local_key: e for e, _ in ents}
    e = by['demo:p1']
    assert e.name == 'Erasmus, Desiderius' and e.variants == ['Erasmus Roterodamus']
    assert e.auth['viaf'] == '95227332' and e.birth == '1466' and e.death == '1536'
    assert by['demo:p2'].name == 'Peter Gast' and by['demo:p2'].auth['wikidata'] == 'Q90'
    pl = by['demo:pl2']
    assert (pl.lat, pl.lon, pl.coord_source) == (47.55, 7.59, 'tei:geo') and pl.auth['geonames'] == '2661604'
    ents2, _ = C.parse_register(root, ctx(), 'p.xml', doc_id='L1')
    assert ents2[0][0].local_key == 'demo:L1#p1'


def test_build_mentions_resolution_and_policy():
    from ratlas.collections.generic_tei import GenericTEI
    c = ctx()
    ms, es = GenericTEI().extract_root(C.parse_xml(LETTER.encode()), rec(), c)
    assert len(es) == 3
    by = {(m.zone, m.role, m.surface): m for m in ms}
    m = by[('hdr', 'addressee', 'Goethe')]
    assert m.local_key == 'demo:L1#p2' and m.in_register and m.auth['gnd'] == '118540238' and m.auth['wikidata'] == 'Q90'
    assert by[('hdr', 'sent_from', 'Paris')].auth['geonames'] == '2988507' and by[('hdr', 'sent_from', 'Paris')].local_key is None
    assert by[('hdr', 'sent_to', 'Basel')].local_key == 'demo:L1#pl2'
    assert all(m.renascor_id == 'demo-entry' and m.licence == 'CC-BY-4.0' and m.tier == 'T1' for m in ms)
    assert all(m.doc_year == 1525 and m.in_period is True and m.date_source == 'letter' for m in ms)
    # T3: no context
    ms3, _ = GenericTEI().extract_root(C.parse_xml(LETTER.encode()), rec(), ctx('T3'))
    assert all(m.context is None for m in ms3)
    # opaque keys
    ms4, _ = GenericTEI().extract_root(C.parse_xml(b'<TEI><text><body><persName ref="#zz9">Z</persName></body></text></TEI>'),
                                       rec('o.xml'), ctx(opaque_keys=True))
    assert ms4[0].local_key == 'demo:zz9' and ms4[0].in_register is False


def test_writer_roundtrip(tmp_path):
    w = C.Writer(tmp_path / 'demo', 'demo', 'CC-BY-NC-4.0', 'T3')
    m = Mention(coll_slug='x', source_path='a.xml', doc_id='a', kind='person', tag='persName', surface='A',
                context='should be dropped', local_key='demo:p', in_register=True, auth={'gnd': '1'}, in_period=True)
    e = Entity(local_key='demo:p', coll_slug='demo', kind='person', name='A', source_path='a.xml', variants=['B'])
    w.add([m, m], [e, e])
    w.note_file('a.xml')
    w.note_file('b.xml', 'boom')
    st = w.close()
    assert st['mentions'] == 2 and st['entities'] == 1 and st['duplicate_entities_dropped'] == 1
    assert st['file_errors'] == 1 and st['resolved_share'] == 1.0 and st['auth_share'] == 1.0
    mt = pq.read_table(tmp_path / 'demo' / 'mentions.parquet')
    assert mt.schema.equals(MENTION_SCHEMA) and mt.column('context').to_pylist() == [None, None]
    et = pq.read_table(tmp_path / 'demo' / 'entities.parquet')
    assert et.schema.equals(ENTITY_SCHEMA) and et.column('n_mentions').to_pylist() == [2]
    assert (tmp_path / 'demo' / 'stats.json').exists()


def test_forbidden_collections():
    assert C.is_forbidden('renascor_2026-10/london-lives-1690-1800/x.xml')
    assert not C.is_forbidden('originals_2026-10-15/web-editions-apis/marinelives/x.xml')


def test_forbidden_tokens_not_overbroad():
    from ratlas import common as C
    assert not C.is_forbidden('bdpn/archivosXMLDocumentos/Lunario_de_Medicina FINAL1.xml')
    assert not C.is_forbidden('originals/dracor_api_tei/ita/de-medici-rappresentazione.xml')
    assert C.is_forbidden('x/london-lives-1690-1800/files/LL.zip!a.xml')
    assert C.is_forbidden('x/Medici Archive Project/a.xml')
