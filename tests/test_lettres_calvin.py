import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS} xml:lang="fr"><teiHeader><fileDesc><sourceDesc><bibl><author><name><surname>Calvin</surname></name></author></bibl></sourceDesc></fileDesc>
<profileDesc><correspDesc><correspAction type="sent"><persName>Jean Calvin</persName>
<settlement key="Genève" ref="https://www.wikidata.org/wiki/Q71" corresp="#geneve">Genève</settlement><date when="1538-01-31"/></correspAction>
<correspAction type="received"><name type='individu' key="Louis du Tillet" ref="https://www.wikidata.org/wiki/Q3263586">Louis du Tillet</name></correspAction>
<correspContext><ref type="prev"><persName></persName></ref></correspContext></correspDesc></profileDesc></teiHeader>
<text><body><p><persName type="collectivité" key="Seigneurs">Seigneurs</persName> à <placeName key="Strasbourg" ref="https://www.wikidata.org/wiki/Q6602">Strasbourg</placeName></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('lettres_calvin', [('x.tar.gz!lettres-calvin-main/src/lib/xml/1538_01.xml', SYN)])
    assert len(ms) == 5
    assert next(m for m in ms if m.surface == 'Seigneurs').kind == 'org'
    assert next(m for m in ms if m.role == 'addressee').auth['wikidata'] == 'Q3263586'
    assert next(m for m in ms if m.role == 'sent_from').auth['wikidata'] == 'Q71'
    assert {m.doc_year for m in ms} == {1538}
    schema_ok(ms)


def test_real_archive_member():
    p = 'renascor_2026-10/lettres-calvin/files/lettres-calvin-main.tar.gz!lettres-calvin-main/src/lib/xml/1538_01_LouisduTillet.xml'
    ms, _ = run_real('lettres_calvin', [p])
    assert count(ms, role='addressee', surface='Louis du Tillet') == 1
    assert count(ms, kind='place', role='sent_to') == 2 and {m.doc_year for m in ms} == {1538}   # v2: 'Envoyée à Paris' is sent_to
    assert sum(m.auth['wikidata'] is not None for m in ms) / len(ms) > 0.7
    schema_ok(ms)
