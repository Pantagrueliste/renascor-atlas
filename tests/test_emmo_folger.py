import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'eng_dh/emmo.zip!emmo/La115.xml'
real = pytest.mark.skipif(not (C.raw_root() / 'eng_dh/emmo.zip').exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>L</title>
<respStmt><name>Michael Poston</name><resp>Encoding Architect</resp></respStmt></titleStmt>
<publicationStmt><date>May, 2017</date></publicationStmt>
<sourceDesc><listBibl><msDesc><history><origin when="1604-04-15">1604 April 15</origin></history></msDesc></listBibl></sourceDesc></fileDesc></teiHeader>
<text><body><p>To <name xml:id="name-001"><w><choice><orig>mr</orig><reg>Mr</reg></choice></w> <w>Allin</w></name> at
<placeName xml:id="name-002"><w><choice><orig>london</orig><reg>London</reg></choice></w></placeName></p></body></text></TEI>'''


def test_synthetic():
    col = get('emmo_folger')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    assert es == [] and [(m.kind, m.tag, m.surface) for m in ms] == [('person', 'name@person', 'mr Allin'), ('place', 'placeName', 'london')]
    assert all(m.doc_year == 1604 and m.date_source == 'header' and m.in_period for m in ms)
    assert all(m.tier == 'T1' for m in ms)                                  # CC BY-SA 4.0 since owner decision of 2026-11-04
    assert all(m.local_key is None and m.ref_raw is None for m in ms)


@real
def test_real():
    col = get('emmo_folger')
    ctx = col.make_context()
    rec = next(r for r in col.iter_files(ctx) if r.path == RAW)
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == rec.count('txt|name', 'txt|placeName')
    assert all(m.doc_year == 1604 for m in ms)
    ms[0].to_row()
