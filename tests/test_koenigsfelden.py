import pytest

from ratlas import common as C
from ratlas import helpers_batch0 as H
from ratlas.collections import koenigsfelden as kf

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

Z = ('renascor_2026-10/koenigsfelden-urkunden-akten/files/koenigsfelden-data-v2024-04-09.zip!'
     'Staatsarchiv-Aargau-koenigsfelden-data-32f6d17/data/docs/')


def test_literal_escape_in_ref_is_stripped():
    xml = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><msDesc><history><origin>'
           '<origDate when="1358-09-12"/></origin></history></msDesc></sourceDesc></fileDesc></teiHeader>'
           '<text><body><p><orgName ref="org002627\\u0009">closters</orgName> '
           '<persName ref="per000001">A</persName></p></body></text></TEI>')
    ctx = C.Context(slug='koenigsfelden', licence='CC-BY-4.0', tier='T1')
    ms, _ = kf.EXTRACTOR.extract_root(C.parse_xml(xml.encode()), C.FileRecord(path='x/y.xml', coll='x'), ctx)
    assert ms[0].ref_raw == 'org002627' and ms[0].doc_year == 1358 and ms[0].in_period is False


def test_real_document_and_registers():
    ms, ents, ctx = H.extract_paths('koenigsfelden', [Z + 'U-17_0282.xml'])
    assert len(ms) > 20 and {m.doc_year for m in ms} == {1358} and {m.in_period for m in ms} == {False}
    assert all(m.zone == 'txt' for m in ms)          # respStmt transcribers are not mentions
    assert any(m.local_key == 'koenigsfelden:per013183' and m.in_register for m in ms)
    by = {e.local_key: e for e in ents}
    assert len(by) == 8863
    kons = by['koenigsfelden:loc000008']
    assert kons.name == 'Konstanz' and kons.auth['hls'] == '7046' and kons.auth['gnd'] == '4032215-4'
    assert abs(kons.lat - 47.66128) < 1e-6 and kons.coord_source == 'tei:geo' and kons.note_type == 'stadt'
    assert by['koenigsfelden:per000089'].birth == '01.05.1218'
    assert by['koenigsfelden:org000030'].note_type == 'zisterzienserkonvent'
