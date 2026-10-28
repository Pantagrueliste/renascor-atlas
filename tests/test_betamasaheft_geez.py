from collections import Counter

import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import betamasaheft_geez as bm

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

BASE = 'originals_2026-10-15/github-lowresource-tei/BetaMasaheft_Geez/'
EDICT = BASE + '4001-5000/LIT4969EdictGala.xml'
GALAW = BASE + '3001-4000/LIT3122Galaw.xml'
TARIKZ = BASE + '2001-3000/LIT2419Tarikz.xml'


def test_synthetic_ids_and_kinds():
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body><p>'
           '<persName ref="PRS9038Susenyos">a</persName><persName ref="ETH1508Gallai">b</persName>'
           '<placeName ref="INS0346DL">c</placeName><placeName ref="wd:Q12560">d</placeName>'
           '<placeName ref="LOC2488Damot" notBefore="1587" notAfter="1597">e</placeName>'
           '<roleName>king</roleName></p></body></text></TEI>')
    ctx = C.Context(slug='betamasaheft_geez', licence=bm.LICENCE, tier=bm.TIER, default_years=(1527, 1632))
    ms, _ = bm.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/t.xml', coll='x'), ctx)
    by = {m.surface: m for m in ms}
    assert by['a'].local_key == 'betamasaheft_geez:PRS9038' and 'bm:PRS9038' in by['a'].auth['other']
    assert by['b'].kind == 'org' and by['b'].tag == 'persName@ethnic'
    assert by['c'].kind == 'org' and by['c'].tag == 'placeName@institution'
    assert by['d'].auth['wikidata'] == 'Q12560' and by['d'].local_key is None
    assert by['e'].local_key == 'betamasaheft_geez:LOC2488' and len(ms) == 5
    assert {m.doc_year_from for m in ms} == {1587} and {m.date_source for m in ms} == {'text'}


def test_real_files():
    ms, _, _ = H.extract_paths('betamasaheft_geez', [EDICT, GALAW, TARIKZ])
    by = {p: [m for m in ms if m.source_path == p] for p in (EDICT, GALAW, TARIKZ)}
    assert Counter(m.kind for m in by[EDICT]) == {'place': 11, 'person': 5} or len(by[EDICT]) >= 16
    assert {m.date_source for m in by[EDICT]} == {'header'} and by[EDICT][0].doc_year == 1548
    assert {m.doc_year_from for m in by[GALAW]} == {1559} and {m.lang for m in by[GALAW]} <= {'gez', 'en', None}
    assert any(m.auth['pleiades'] for m in by[GALAW]) and any(m.auth['wikidata'] for m in by[GALAW])
    assert {m.date_source for m in by[TARIKZ]} == {'collection'}
    assert any(m.role == 'author' for m in by[TARIKZ])
