import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><title>T</title><author key="Anónimo">Anónimo</author>
<editor>Daria Castaldo</editor></titleStmt><publicationStmt><date when="2016"/></publicationStmt></fileDesc>
<profileDesc><creation><date when="1613"/></creation></profileDesc></teiHeader>
<text><body><p><persName type="authority" key="Faria e Sousa, Manuel de">Faria</persName> <name type="polemista">Lope</name>
<persName type="polemist" key="Góngora, Luis de">Góngora</persName> <name>vuesa merced</name></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('polemicagongorina_obvil', [('polemos/1613_x.xml', SYN)])
    assert len(ms) == 3                              # anonymous author and untyped name skipped
    f = next(m for m in ms if m.surface == 'Faria')
    assert f.tag == 'persName@authority' and f.local_key == 'polemicagongorina_obvil:Faria e Sousa, Manuel de'
    assert next(m for m in ms if m.surface == 'Lope').tag == 'name@polemista'
    assert {m.doc_year for m in ms} == {1613} and all(m.licence == 'LicenseRef-CC-BY-NC-ND-3.0-FR' for m in ms)
    schema_ok(ms)


def test_real_file():
    ms, _ = run_real('polemicagongorina_obvil', ['polemos/1613_carta-amigo.xml', 'polemos/1617_pasajero-figueroa.xml'])
    assert count(ms, tag='persName@authority') >= 14 and count(ms, tag='persName@polemist') >= 1
    assert count(ms, tag='author') == 1                 # the anonymous author of 1613_carta-amigo is skipped
    assert count(ms, local_key='polemicagongorina_obvil:Virgilio') >= 2 and all(len(m.context or '') <= 120 for m in ms)
    schema_ok(ms)
