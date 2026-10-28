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
    xml = """<TEI><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><country>Espana</country><settlement>Madrid</settlement></msIdentifier>
    <history><origin><origDate when="1554">1554</origDate><origPlace><placeName>Milan</placeName><region type="province"/><country>Italia</country><geo>45.466944, 9.19</geo></origPlace></origin></history></msDesc></sourceDesc></fileDesc>
    <profileDesc><correspDesc><correspAction type="sent"><placeName type="origin"><location>Italia, Milan</location><geo>N45\u00b027'51.37" E9\u00b011'22.24"</geo></placeName>
    <persName><name>Bernardo de Bolea</name><gender value="H">hombre</gender><socecStatus>Noble</socecStatus></persName></correspAction>
    <correspAction type="received"><placeName type="destination"><location>, </location><geo/></placeName><persName><name>Granvela</name><gender value="H">hombre</gender></persName></correspAction></correspDesc></profileDesc></teiHeader>
    <text><body><p><tok pos="NP00000">Milan</tok></p></body></text></TEI>"""
    ms, es = H.synthetic('h15corpus', xml)
    r = {(m.role, m.surface): m for m in ms}
    assert ('sender', 'Bernardo de Bolea') in r and ('addressee', 'Granvela') in r      # <name>, not gender/status labels
    assert ('sent_from', 'Italia, Milan') in r and r[('sent_from', 'Italia, Milan')].in_register is True
    assert ('repository', 'Madrid') in r and not any(k[1] in ('Espana', 'Italia') for k in r)   # qualifiers dropped
    assert not any(m.surface == 'Milan' for m in ms)               # origPlace repeats the correspAction origin; text toks not read
    assert len(es) == 1 and abs(es[0].lat - 45.464269) < 1e-5 and abs(es[0].lon - 9.189511) < 1e-5     # DMS of the correspAction geo
    assert r[('sender', 'Bernardo de Bolea')].tier == 'T3' and r[('sender', 'Bernardo de Bolea')].context is None


def test_dms():
    lat, lon = H.parse_coord("N45\u00b027'51.37\" E9\u00b011'22.24\"")
    assert abs(lat - 45.464269) < 1e-5 and abs(lon - 9.189511) < 1e-5
    assert H.parse_coord('') is None and H.parse_coord('999, 9') is None


def test_real_file_repeated_header():
    ms, es = H.extract_paths('h15corpus', 'GRANVELA-0004.xml')
    assert len(ms) == 4 and len(es) == 1 and es[0].coord_source == 'tei:geo'     # 3 copies of the header, one read
    assert {m.role for m in ms} == {'sender', 'addressee', 'sent_from', 'repository'}
    assert {m.doc_year for m in ms} == {1547}
    _rows_ok(ms)
