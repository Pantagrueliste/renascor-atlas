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
    xml = f"""<TEI {NS}><text><body><l><persName ref="#charlemagne">Charles</persName> <persName ref="dieu">Dieu</persName> <persName>sans ref</persName>
    <placeName ref="#troie">Troye</placeName> <orgName>les Francs</orgName></l></body></text></TEI>"""
    ms, _ = H.synthetic('elan_crenum', xml)
    r = {m.surface: m for m in ms}
    assert r['Charles'].local_key == 'elan_crenum:charlemagne' and r['Dieu'].local_key == 'elan_crenum:dieu'   # ref without '#'
    assert r['sans ref'].local_key is None and r['les Francs'].kind == 'org'
    assert r['Troye'].doc_year == 1515 and r['Troye'].date_source == 'collection' and r['Troye'].in_period is True


def test_real_file():
    ms, _ = H.extract_paths('elan_crenum', 'data__xml__books__Livre_I.xml')
    assert len(ms) > 800 and {m.kind for m in ms} == {'person', 'place', 'org'}
    assert any(m.local_key == 'elan_crenum:dieu' for m in ms)
    _rows_ok(ms)
