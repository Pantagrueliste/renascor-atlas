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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><title>Tax <placeName>Bristol</placeName> <date when="1670">1670</date></title></titleStmt>
    <sourceDesc><msDesc><msIdentifier><settlement>London</settlement></msIdentifier></msDesc></sourceDesc></fileDesc></teiHeader>
    <text><body><div><head><placeName type="ward"><choice><orig>Ward S<ex>an</ex>cti</orig></choice><reg>Ward of St Michael</reg></placeName></head>
    <p><persName ana="h_tp" ref="https://gams.uni-graz.at/o:htx.persons#P12">John Smith</persName><measure unit="hearths">3</measure></p>
    <p><persName ana="h:owner">Thomas Ball</persName> et <persName ana="h:tenant">tenen</persName></p></div></body></text></TEI>"""
    ms, _ = H.synthetic('hearth_tax', xml)
    by = {(m.kind, m.role, m.surface): m for m in ms}
    assert ('place', 'mention', 'Ward of St Michael') in by          # regularised reading
    assert by[('place', 'mention', 'Ward of St Michael')].tag == 'placeName@ward'
    jt = by[('person', 'taxpayer', 'John Smith')]
    assert jt.local_key == 'hearth_tax:person#P12' and jt.in_register is False
    assert not any(x.startswith('url:') for x in jt.auth['other'])
    assert ('person', 'owner', 'Thomas Ball') in by and ('person', 'tenant', 'tenen') in by
    assert ('place', 'repository', 'London') in by
    assert jt.doc_year == 1670 and jt.in_period is True
    _rows_ok(ms)


def test_real_file():
    ms, es = H.extract_paths('hearth_tax', 'o:htx.WM2.xml')
    assert len(ms) == 467 and es == []
    assert sum(m.role == 'taxpayer' for m in ms) == 449
    assert {m.date_source for m in ms} == {'header'} and all(m.in_period for m in ms)
    assert {m.tier for m in ms} == {'T2'} and {m.licence for m in ms} == {'CC-BY-NC-4.0'}
    _rows_ok(ms)
