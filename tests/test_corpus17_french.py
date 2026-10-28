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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><author ref="isni:0000000121296128">Corneille, Pierre</author><editor ref="orcid:0000-0001-9094-4475">Simon</editor></titleStmt>
    <sourceDesc><bibl><author ref="isni:0000000121296128">Corneille, Pierre</author><publisher ref="isni:0000000416536993">Targa, Francois</publisher>
    <pubPlace ref="geonames:2988507">Paris</pubPlace><date when="1639">1639</date></bibl></sourceDesc></fileDesc></teiHeader>
    <text><body><p>chez <persName>FRANCOIS</persName> a <placeName>PARIS</placeName></p></body></text></TEI>"""
    ms, _ = H.synthetic('corpus17_french', xml, path='corpus17/data/EDI_0002.xml')
    r = {(m.role, m.surface): m for m in ms}
    assert len(ms) == 5                                               # author once, editor never
    assert r[('author', 'Corneille, Pierre')].auth['other'] == ['isni:0000000121296128']
    assert r[('pubplace', 'Paris')].auth['geonames'] == '2988507' and r[('publisher', 'Targa, Francois')].kind == 'person'
    assert {m.doc_year for m in ms} == {1639}


def test_only_full_files():
    from ratlas.collections import get
    col = get('corpus17_french')
    fs = col.iter_files(col.make_context())
    assert fs and all(r.name.startswith('EDI_') and r.name[4:8].isdigit() and r.name.endswith('.xml')
                      and '-' not in r.name and 'header' not in r.name for r in fs)
    assert len(fs) == 15


def test_real_file():
    ms, _ = H.extract_paths('corpus17_french', 'EDI_0001.xml')
    assert len(ms) == 24 and sum(m.role == 'pubplace' for m in ms) == 1
    assert {m.doc_year for m in ms} == {1624}
    _rows_ok(ms)
