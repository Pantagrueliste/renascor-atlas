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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><title>x</title></titleStmt><publicationStmt><date when="1798-01-18">1798</date></publicationStmt></fileDesc></teiHeader>
    <text><body><p><rs type="person" key="PvCo">Prinzen von Conde</rs> <rs type="person" key="PvCo">er</rs> <rs type="place" key="Pe">Petersburg</rs>
    <rs type="god" key="Zeus">Jupiter</rs> <rs type="ship" key="S1">Die Maria</rs> <name type="place" key="Hb">Hamburg</name>
    <rs type="place" key="Pe">dahier</rs></p></body></text></TEI>"""
    ms, _ = H.synthetic('germanc', xml, path='de_dh/germanc.zip!TEI/NEWS_P3_NoD_1750_x.xml')
    k = {m.surface: m for m in ms}
    assert 'Die Maria' not in k                                     # ships are not atlas entities
    assert k['Prinzen von Conde'].role == 'mention' and k['er'].role == 'anaphor' and k['dahier'].role == 'anaphor'
    assert k['Prinzen von Conde'].local_key == 'germanc:NEWS_P3_NoD_1750_x#PvCo'      # text-local key
    assert k['Jupiter'].kind == 'person' and k['Jupiter'].tag == 'rs@god'
    assert k['Hamburg'].tag == 'name@place'
    assert k['Petersburg'].doc_year == 1750 and k['Petersburg'].date_source == 'filename' and not k['Petersburg'].in_period


def test_real_file():
    ms, _ = H.extract_paths('germanc', 'TEI/NEWS_P3_NoD_1798_danzig.xml')
    assert len(ms) == 251 and sum(m.role == 'anaphor' for m in ms) == 72
    assert all(m.doc_year == 1798 and m.in_period is False for m in ms)
    assert sum(m.local_key is not None for m in ms) == 250
    _rows_ok(ms)


def test_only_tei_folder():
    from ratlas.collections import get
    col = get('germanc')
    fs = col.iter_files(col.make_context())
    assert fs and all('!TEI/' in r.path for r in fs)
