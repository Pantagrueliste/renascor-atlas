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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><sourceDesc><msDesc><msIdentifier><country>Belgien</country><settlement>Brussel</settlement></msIdentifier>
    <history><origin><origDate when="1622-04-29">29. April 1622</origDate></origin></history></msDesc></sourceDesc></fileDesc></teiHeader>
    <text><body><p><rs type="person" ref="#rubens_peter_paul">Rubens</rs> <rs type="place" ref="#antwerpen">Anvers</rs>
    <rs type="corporate" ref="#gilde">gilde</rs> <rs type="work" ref="#jaffe_189">tableau</rs></p></body></text></TEI>"""
    ms, _ = H.synthetic('rubens_online', xml)
    r = {m.surface: m for m in ms}
    assert 'tableau' not in r                                         # works are not persons/places/orgs
    assert r['Rubens'].local_key == 'rubens_online:rubens_peter_paul' and r['gilde'].kind == 'org'
    assert r['Brussel'].role == 'repository' and r['Belgien'].role == 'repository'
    assert r['Rubens'].tier == 'T3' and r['Rubens'].context is None  # T3: no snippets
    assert r['Rubens'].doc_year == 1622


def test_viten_filename_date():
    ms, _ = H.extract_paths('rubens_online', 'viten/Bullart-Academie-des-Sciences-1682.xml')
    assert ms and {m.doc_year for m in ms} == {1682} and {m.date_source for m in ms} == {'filename'}
    assert {m.kind for m in ms} >= {'person', 'place'} and all(m.context is None for m in ms)
    _rows_ok(ms)


def test_real_archival():
    ms, _ = H.extract_paths('rubens_online', 'AGRB_papiers-etat-1622.xml')
    assert len(ms) == 5 and {m.doc_year for m in ms} == {1622}
    assert sum(m.role == 'repository' for m in ms) == 2
