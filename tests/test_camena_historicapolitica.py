import pytest

from b5util import NS, REAL, count, run, run_real, schema_ok

pytestmark = pytest.mark.skipif(not REAL, reason='raw data not available')

SYN = b'''<TEI.2><teiHeader><fileDesc><titleStmt><title>T</title><author>Bose</author></titleStmt>
<sourceDesc><bibl>Basel: [Guarin], 1581.</bibl></sourceDesc></fileDesc>
<revisionDesc><change><date>12/2007</date><respStmt><name>Reinhard Gruhl</name></respStmt></change></revisionDesc></teiHeader>
<text><body><p><name type='person'>Petrus Bembus</name> <name type='myth'>Iuppiter</name></p></body></text></TEI.2>'''


def test_synthetic():
    ms, _ = run('camena_historicapolitica', [('originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/historicapolitica/x.xml', SYN)])
    assert [(m.tag, m.surface, m.doc_year, m.date_source) for m in ms] == [
        ('name@person', 'Petrus Bembus', 1581, 'header'), ('name@myth', 'Iuppiter', 1581, 'header')]
    schema_ok(ms)


def test_real_file():
    p = 'originals_2026-10-15/github-neolatin-french-tei/camena-neolatinlit/historicapolitica/Pantaleon_res_memorabiles_1.xml'
    ms, _ = run_real('camena_historicapolitica', [p])
    assert len(ms) == 22 and {m.doc_year for m in ms} == {1581} and all(m.in_period for m in ms)
