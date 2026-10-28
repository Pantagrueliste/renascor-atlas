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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><author>Eleonora</author></titleStmt><sourceDesc><msDesc><msIdentifier><settlement>Munchen</settlement></msIdentifier></msDesc></sourceDesc></fileDesc>
    <profileDesc><correspDesc><correspAction type="sent"><persName ref="#emt_person_id__9">Eleonora</persName><placeName ref="#emt_place_id__63">Wien</placeName><date when-iso="1685-03-01">1. Marz 1685</date></correspAction>
    <correspAction type="received"><persName ref="#emt_person_id__18">Johann Wilhelm</persName><placeName ref="#"/><date/></correspAction></correspDesc></profileDesc></teiHeader>
    <text><body><p><rs type="person" ref="#emt_person_id__25">Kaiser</rs> <rs type="place" ref="#emt_place_id__73">Schpanien</rs></p></body></text></TEI>"""
    ms, _ = H.synthetic('kaiserin_eleonora_magdalena', xml)
    r = {(m.role, m.surface): m for m in ms}
    assert ('sender', 'Eleonora') in r and ('addressee', 'Johann Wilhelm') in r
    assert ('sent_from', 'Wien') in r and ('repository', 'Munchen') in r
    assert len([m for m in ms if m.kind == 'place' and m.role == 'sent_to']) == 0      # empty ref="#" tag dropped
    assert r[('mention', 'Kaiser')].local_key == 'kaiserin_eleonora_magdalena:emt_person_id__25'
    assert r[('sender', 'Eleonora')].doc_year == 1685 and r[('sender', 'Eleonora')].date_source == 'letter'
    assert r[('sender', 'Eleonora')].in_period is True and r[('sender', 'Eleonora')].tier == 'T1'


def test_real_file():
    ms, _ = H.extract_paths('kaiserin_eleonora_magdalena', 'kasten_blau_44_10_0068.xml')
    assert len(ms) == 59
    assert {m.role for m in ms if m.zone == 'hdr' and m.kind == 'person'} >= {'sender', 'addressee'}
    assert all(m.doc_year == 1709 and m.in_period is False for m in ms)       # a 1709 letter: outside 1450-1700
    assert sum(m.local_key is not None for m in ms) == 57
    _rows_ok(ms)
