import pytest

from ratlas import common as C
from ratlas import helpers_batch1 as H

pytestmark = pytest.mark.skipif(not (C.raw_root().exists() and C.scan_path().exists()),
                                reason='raw data / scan not available')

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def _rows_ok(ms):
    for m in ms[:50]:
        r = m.to_row()
        assert r['kind'] in ('person', 'place', 'org') and r['part'] in ('txt', 'hdr')
        assert set(r['auth']) >= {'gnd', 'viaf', 'wikidata', 'geonames', 'other'}
        if r['context']:
            assert len(r['context']) <= 120


def test_synthetic():
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><respStmt><persName>Editor Modern</persName></respStmt></titleStmt>
    <sourceDesc><msDesc><msIdentifier><settlement>Bordeaux</settlement></msIdentifier></msDesc><biblStruct><monogr><author><persName key="pers1">Vinet</persName></author>
    <imprint><date when="1583">1583</date></imprint></monogr></biblStruct>
    <listPerson><person xml:id="antoine_loisel"><persName>Antoine Loisel</persName></person><person xml:id="oronce_fine"><persName>Oronce Fine</persName></person></listPerson></sourceDesc></fileDesc></teiHeader>
    <text><body><p><name key="antoine_loisel">Monsieur Loisel</name> et <name>Orontius</name> et <name corresp="#oronce_fine">Orontius</name>
    <persName>Dieu</persName> a <placeName type="ville"><choice><orig>BOVRDEAVS</orig><reg>BOURDEAUS</reg></choice></placeName></p></body></text></TEI>"""
    ms, es = H.synthetic('freem_bvh', xml)
    assert len(es) == 2
    r = [(m.kind, m.tag, m.surface, m.local_key) for m in ms]
    assert ('person', 'name', 'Monsieur Loisel', 'freem_bvh:doc#antoine_loisel') in r      # untyped name resolved by the register
    assert ('person', 'name', 'Orontius', 'freem_bvh:doc#oronce_fine') in r
    assert sum(1 for x in r if x[2] == 'Orontius') == 1          # the untyped name without reference is skipped
    assert ('place', 'placeName', 'BOVRDEAVS', None) in r                                  # diplomatic reading
    assert not any(x[2] == 'Editor Modern' for x in r)
    assert {m.doc_year for m in ms} == {1583} and {m.tier for m in ms} == {'T2'}


def test_real_file():
    ms, es = H.extract_paths('freem_bvh', 'B330636101_D55869_tei.xml')
    assert len(ms) == 17 and len(es) == 8
    assert sum(m.in_register is True for m in ms) == 10 and {m.doc_year for m in ms} == {1583}
    _rows_ok(ms)
