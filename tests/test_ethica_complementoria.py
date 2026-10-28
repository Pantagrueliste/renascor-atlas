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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><publicationStmt><availability><licence target="http://creativecommons.org/licenses/by-nc/3.0/de/"/></availability></publicationStmt></fileDesc>
    <profileDesc><particDesc><listPerson><person xml:id="SE"><birth>1600</birth></person></listPerson></particDesc></profileDesc></teiHeader>
    <text><body><p><name type="person" key="pnd118718622">Georg</name> <name type="person" ref="#SE">Susanna</name>
    <name type="person" ref="pnd:0000">anon</name> <name>untyped</name> <persName ref="http://d-nb.info/gnd/XXX">G</persName></p></body></text></TEI>"""
    ms, es = H.synthetic('ethica_complementoria', xml, path='queue5/ethica.zip!EthicaComplementoria-master/transcriptions/a/Foo 1648.xml')
    r = {m.surface: m for m in ms}
    assert 'untyped' not in r
    assert r['Georg'].auth['gnd'] == '118718622'
    assert r['Susanna'].in_register is True and r['Susanna'].local_key.endswith('#SE')
    assert r['anon'].auth['gnd'] is None and r['G'].auth['other'] == []                 # placeholder ids removed
    assert {m.licence for m in ms} == {'CC-BY-NC-3.0-DE'} and {m.tier for m in ms} == {'T2'}
    assert {m.doc_year for m in ms} == {1648} and {m.date_source for m in ms} == {'filename'}


def test_real_file():
    ms, es = H.extract_paths('ethica_complementoria', 'Klag und Trost_1652.xml')
    assert len(ms) == 6 and len(es) == 3
    assert {m.doc_year for m in ms} == {1652}
    assert {m.tier for m in ms} == {'T1'}
    _rows_ok(ms)


def test_no_articles():
    from ratlas.collections import get
    col = get('ethica_complementoria')
    fs = col.iter_files(col.make_context())
    assert fs and not any('Aufsatz' in r.path for r in fs)
