import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = f'''<TEI {NS}><teiHeader><fileDesc><titleStmt><title>T</title>
<author ref="http://www.viaf.org/viaf/36914132">Balde, Iacob<date>1604-1668</date></author></titleStmt>
<publicationStmt><date>2004</date></publicationStmt></fileDesc></teiHeader>
<text><body><p><name type="person">Iacobus</name> <name type="myth">Iuppiter</name> <name type="nation">Arabes</name>
<rs type="person" n="Vergilius">poeta</rs> <placeName key="Roma" ref="http://www.viaf.org/2477678">Romae</placeName>
<name>untyped</name></p></body></text></TEI>'''.encode()


def test_synthetic():
    ms, ents = run('corpus_corporum', [('corpuscorp/xml/9.xml', SYN)])
    assert ents == []
    by = {(m.kind, m.tag) for m in ms}
    assert ('person', 'name@myth') in by and ('org', 'name@nation') in by and ('person', 'author') in by
    a = next(m for m in ms if m.tag == 'author')
    assert a.role == 'author' and a.surface == 'Balde, Iacob' and a.auth['viaf'] == '36914132'
    assert next(m for m in ms if m.tag == 'rs@person').ref_raw == 'n:Vergilius'
    assert next(m for m in ms if m.tag == 'placeName').auth['viaf'] == '2477678'
    assert len(ms) == 6                                     # the untyped <name> is skipped
    assert {m.doc_year for m in ms} == {1668} and {m.date_source for m in ms} == {'header'}   # author death proxy
    schema_ok(ms)


def test_real_file():
    ms, _ = run_real('corpus_corporum', ['corpuscorp/xml/13978.xml'])
    assert count(ms, kind='place') > 200 and count(ms, tag='name@myth') > 100
    assert all(m.doc_year == 1579 for m in ms)
    schema_ok(ms)
