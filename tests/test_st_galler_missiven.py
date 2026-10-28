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
    xml = f"""<TEI {NS}><teiHeader><profileDesc><correspDesc><correspAction type="sent"><orgName ref="stadtasg-actors-56">Konstanz</orgName><date>1453-08-20</date></correspAction>
    <correspAction type="received"><orgName ref="stadtasg-actors-148">St. Gallen</orgName><placeName ref="stadtasg-places-38">St. Gallen</placeName></correspAction></correspDesc></profileDesc></teiHeader>
    <text><body><p><persName ref="stadtasg-actors-453">Peterman</persName> in <placeName ref="stadtasg-places-102">Wil</placeName>
    <orgName ref="stadtasg-actors-56" role="issuer">Burgermaister</orgName> <orgName ref="stadtasg-actors-148" role="recipient">rate</orgName></p></body></text></TEI>"""
    ms, _ = H.synthetic('st_galler_missiven', xml)
    r = {(m.zone, m.role, m.surface): m for m in ms}
    assert ('hdr', 'sender', 'Konstanz') in r and ('hdr', 'addressee', 'St. Gallen') in r and ('hdr', 'sent_to', 'St. Gallen') in r
    assert ('txt', 'signer', 'Burgermaister') in r and ('txt', 'addressee', 'rate') in r
    p = r[('txt', 'mention', 'Peterman')]
    assert p.local_key == 'st_galler_missiven:stadtasg-actors-453' and p.in_register is False and p.doc_year == 1453


def test_real_file():
    ms, _ = H.extract_paths('st_galler_missiven', 'stasg_missiv_00289.xml')
    assert len(ms) == 10 and all(m.local_key for m in ms)
    assert {m.doc_year for m in ms} == {1453} and all(m.in_period for m in ms)
    assert {m.role for m in ms} >= {'sender', 'addressee', 'signer', 'sent_to'}
    assert {m.tier for m in ms} == {'T2'}
    _rows_ok(ms)
