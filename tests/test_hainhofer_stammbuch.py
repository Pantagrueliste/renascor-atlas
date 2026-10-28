import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'originals_2026-10-15/oct-binary/oct__hainhofer-grosses-stammbuch/files/stammbuch-18.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>x</title>
<respStmt><persName ref="https://d-nb.info/gnd/118700804">Philipp Hainhofer</persName></respStmt></titleStmt></fileDesc></teiHeader>
<text><body><div type="Eintrag" xml:id="e1"><div type="Datierung"><p><date notBefore-iso="1610" notAfter-iso="1612">1610</date></p></div>
<div type="Name"><p><rs type="person" ref="psn:a_b">A B</rs></p></div>
<div type="Deutung"><p><rs type="person" ref="psn:c_d">C D</rs></p></div></div>
<div type="Eintrag" xml:id="e2"><div type="Datierung"><p><date when-iso="1905">1905</date></p></div>
<div type="Deutung"><p><rs type="person" ref="psn:c_d">C D</rs></p></div></div>
<div type="Eintrag" xml:id="e3"><div type="Deutung"><p><rs type="person" ref="psn:e_f">E</rs></p></div></div></body></text></TEI>'''


def test_synthetic():
    col = get('hainhofer_stammbuch')
    ctx = col.make_context()
    ms, es = col.extractor.extract_root(C.parse_xml(SYNTH), C.FileRecord(path=RAW, coll='x'), ctx)
    assert es == [] and len(ms) == 4                                   # respStmt person is a modern editor: skipped
    a, c1, c2, e = ms
    assert a.role == 'author' and a.local_key == 'hainhofer_stammbuch:a_b' and a.in_register is False
    assert a.zone_class is None and c1.zone_class == 'editorial_commentary' and e.zone_class == 'editorial_commentary'
    assert (c1.doc_year_from, c1.doc_year_to, c1.in_period, c1.date_source) == (1610, 1612, True, 'header')
    assert c2.in_period is False and c2.doc_year == 1905
    assert (e.doc_year_from, e.doc_year_to, e.date_source) == (1596, 1633, 'collection')


@real
def test_real():
    col = get('hainhofer_stammbuch')
    ctx = col.make_context()
    files = col.iter_files(ctx)
    ctx.all_files = files
    col.prepare(ctx)
    rec = next(iter(files))
    ms, _ = col.extract(rec, ctx)
    from ratlas import classify
    for m in ms:
        classify.finalize(m)
    rs = [m for m in ms if m.role != 'written_at']
    assert len(rs) == rec.count('txt|rs') == 1486
    assert len(ms) == 1486 + 103                                         # v2: + the 'Ort' div of 103 entries
    assert len({m.local_key for m in rs}) == 201 and sum(m.in_register is True for m in rs) >= 1480  # 1 of 201 slugs is not in the register
    assert sum(m.role == 'author' for m in ms) == 108
    # v2: ~80% of the person tags are the modern commentary of each entry (excluded from the default counts)
    comm = sum(m.zone_class == 'editorial_commentary' for m in rs)
    assert 1150 < comm < 1260
    ort = [m for m in ms if m.role == 'written_at']
    assert all(m.kind == 'place' for m in ort) and sum(m.cert == 'low' for m in ort) > 30
    assert sum(m.in_register is True for m in ort) > 70                  # 81 of the 103 match a register place
    assert 0.95 < sum(bool(m.in_period) for m in ms) / len(ms) < 0.99
    assert max(len(m.context or '') for m in ms) <= 120
    ms[0].to_row()
