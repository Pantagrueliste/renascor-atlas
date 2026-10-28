import pytest

from ratlas import common as C
from ratlas.collections import get

BASE = 'renascor_2026-10/philipp-hainhofer-reiseberichte-sammlungsbeschreibungen-1594-1636-annotated-digital-edition/files/'
real = pytest.mark.skipif(not (C.raw_root() / BASE / 'Dresden_1629.xml').exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>T</title>
<respStmt><name>Projekt</name></respStmt></titleStmt><publicationStmt><date when="2026"/></publicationStmt></fileDesc></teiHeader>
<text><body><p><rs type="person" role="remote" ref="psn:a_b">A</rs> in <rs type="place" ref="plc:augsburg">Augsburg</rs>
sah <rs type="work" ref="wrk:kunstschrank">Schrank</rs> der <rs type="org" ref="org:reichstag">Reichstag</rs>
<rs type="literature" ref="lit:x_1603">x</rs></p></body></text></TEI>'''


def test_synthetic():
    col = get('hainhofer_reisen')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=BASE + 'Stuttgart_1621.xml', coll='x'), ctx)
    assert es == [] and [(m.kind, m.local_key) for m in ms] == [
        ('person', 'hainhofer_reisen:a_b'), ('place', 'hainhofer_reisen:augsburg'), ('org', 'hainhofer_reisen:reichstag')]
    assert all(m.doc_year == 1621 and m.date_source == 'filename' and m.in_register is False for m in ms)


@real
def test_real():
    col = get('hainhofer_reisen')
    ctx = col.make_context()
    rec = next(r for r in col.iter_files(ctx) if r.path.endswith('Dresden_1629.xml'))
    ms, _ = col.extract(rec, ctx)
    assert len(ms) <= rec.count('txt|rs')      # work/literature tags are skipped (none on this file)
    assert sum(m.kind == 'person' for m in ms) == 629
    assert all(m.doc_year == 1629 and m.in_period for m in ms)
    ms[0].to_row()
