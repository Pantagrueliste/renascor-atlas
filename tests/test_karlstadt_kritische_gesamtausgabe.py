import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><title>De X <date when="1507-08-10">1507</date></title>
<author ref="#karlstadt_andreas">Andreas Bodenstein von Karlstadt</author></titleStmt>
<publicationStmt><date type="digitised" when="2013"/></publicationStmt></fileDesc></teiHeader>
<text><body><p><rs type="person" ref="#thomas_aquinas">Thome</rs> <rs type="place" ref="#wittenberg">Wittenberg</rs></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, _ = run('karlstadt_kritische_gesamtausgabe', [('renascor_2026-10/karlstadt-kritische-gesamtausgabe/files/xml/001_transcript.xml', SYN)])
    assert len(ms) == 3 and {m.doc_year for m in ms} == {1507}
    a = next(m for m in ms if m.tag == 'author')
    assert a.role == 'author' and a.local_key == 'karlstadt_kritische_gesamtausgabe:karlstadt_andreas'
    assert count(ms, kind='place', local_key='karlstadt_kritische_gesamtausgabe:wittenberg') == 1
    assert all(m.tier == 'T3' and m.context is None for m in ms)
    # v2: the register of the edition (ed000216/register) resolves the slugs: names, GND, GeoNames
    th = next(m for m in ms if m.surface == 'Thome')
    assert th.in_register is True and th.auth['gnd']
    assert next(m for m in ms if m.kind == 'place').auth['geonames']
    schema_ok(ms)


def test_real_file():
    ms, _ = run_real('karlstadt_kritische_gesamtausgabe', ['renascor_2026-10/karlstadt-kritische-gesamtausgabe/files/xml/001_transcript.xml'])
    assert count(ms, kind='person') > 600 and count(ms, kind='place') >= 20
    assert count(ms, local_key='karlstadt_kritische_gesamtausgabe:thomas_aquinas') > 5
    assert {m.doc_year for m in ms} == {1507}
