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
    xml = f"""<TEI {NS}><teiHeader><fileDesc><titleStmt><title>t</title><editor><persName>Modern Editor</persName></editor></titleStmt>
    <sourceDesc><msDesc><history><origin><origDate when="1637-07-30"/></origin></history></msDesc></sourceDesc></fileDesc></teiHeader>
    <text><body><p>Im <placeName ref="loc007683.02">bosen thurn</placeName> <persName ref="per009462a.01">Gribollet</persName>
    <persName>ohne ref</persName> <orgName ref="org004722.02">Rat</orgName></p></body></text></TEI>"""
    ms, _ = H.synthetic('ssrq_swisslaw', xml)
    assert len(ms) == 4 and all(m.zone == 'txt' for m in ms)          # the editor in the header is not read
    k = {m.surface: m for m in ms}
    assert k['Gribollet'].local_key == 'ssrq_swisslaw:per009462a' and k['Gribollet'].in_register is False   # v2: '.01' variant suffix stripped
    assert k['Gribollet'].ref_raw == 'per009462a.01'
    assert k['ohne ref'].local_key is None
    assert k['Rat'].kind == 'org' and k['bosen thurn'].kind == 'place'
    assert k['Rat'].doc_year == 1637 and k['Rat'].licence == 'CC-BY-NC-SA-4.0' and k['Rat'].tier == 'T2'


def test_real_file():
    ms, _ = H.extract_paths('ssrq_swisslaw', 'SSRQ-FR-I_2_8-100.5-1.xml')
    assert len(ms) == 35 and sum(m.local_key is not None for m in ms) == 18
    assert {m.doc_year for m in ms} == {1637} and all(m.in_period for m in ms)
    assert any(m.local_key == 'ssrq_swisslaw:per029440a' and m.ref_raw == 'per029440a.01' for m in ms)
    _rows_ok(ms)


def test_variant_suffix_one_entity_and_principal_date():
    xml = f"""<TEI {NS}><teiHeader><fileDesc><sourceDesc><msDesc><msContents><msItem>
    <filiation type="original"><origDate from="1395-01-01" to="1395-12-31">Undatiert</origDate></filiation>
    <filiation type="copy"><origDate when="1619">Abschrift</origDate></filiation></msItem></msContents>
    <history><origin><origDate when="1619"/></origin></history></msDesc></sourceDesc></fileDesc></teiHeader>
    <text><body><p><placeName ref="loc007693.04">Staffis</placeName> <placeName ref="loc007693.12">Estavayer</placeName></p></body></text></TEI>"""
    ms, _ = H.synthetic('ssrq_swisslaw', xml)
    assert {m.local_key for m in ms} == {'ssrq_swisslaw:loc007693'}
    assert {m.surface for m in ms} == {'Staffis', 'Estavayer'}
    # the original (1395), not the copy (1619): a single year, not 1395-1619
    assert {(m.doc_year_from, m.doc_year_to, m.doc_year) for m in ms} == {(1395, 1395, 1395)}
    assert all(m.in_period is False for m in ms)
