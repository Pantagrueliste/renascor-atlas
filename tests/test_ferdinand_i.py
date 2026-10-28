import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'originals_2026-10-15/oct-binary/oct__ferdinand-i-familienkorrespondenz/files/A001.xml'
real = pytest.mark.skipif(not (C.raw_root() / RAW).exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt><title>A1.</title>
<author><persName ref="nognd">Karl</persName></author></titleStmt>
<sourceDesc><biblFull><titleStmt><author><persName>Ferdinand I., Kaiser</persName></author></titleStmt></biblFull></sourceDesc></fileDesc>
<profileDesc><correspDesc><correspAction type="sent"><persName ref="nognd">Karl</persName><placeName>Wien</placeName>
<date when="1520-03-01">1520 M\xc3\xa4rz 1</date></correspAction>
<correspAction type="received"><persName ref="nognd">Ferdinand</persName></correspAction></correspDesc></profileDesc></teiHeader>
<text><body><div type="letter"><p>In <placeName ref="2761369" key="Wien">Wien</placeName> sah <persName ref="nognd" key="Orti, Gabriel von">Horti</persName>.</p></div></body></text></TEI>'''


def _run(data, name='A999.xml'):
    col = get('ferdinand_i')
    ctx = col.make_context()
    ctx.options['_gn_by_name'] = {'Wien': '2761369'}
    root = C.parse_xml(data)
    return col.extractor.extract_root(root, C.FileRecord(path=RAW.replace('A001.xml', name), coll='x'), ctx)[0]


def test_synthetic():
    ms = _run(SYNTH)
    by = {(m.role, m.surface) for m in ms}
    assert ('sender', 'Karl') in by and ('addressee', 'Ferdinand') in by and ('sent_from', 'Wien') in by
    assert not any(m.role == 'author' for m in ms)                     # duplicate of the sender / principal dropped
    assert len(ms) == 5
    sf = next(m for m in ms if m.role == 'sent_from')
    assert sf.auth['geonames'] == '2761369' and sf.local_key == 'ferdinand_i:gn2761369'   # id by name from the text tag
    txt = next(m for m in ms if m.zone == 'txt' and m.kind == 'person')
    assert txt.local_key == 'ferdinand_i:Orti, Gabriel von' and txt.in_register is False
    assert all(m.doc_year == 1520 and m.date_source == 'letter' for m in ms)


@real
def test_real_letter():
    col = get('ferdinand_i')
    ctx = col.make_context()
    rec = next(r for r in col.iter_files(ctx) if r.path == RAW)
    ctx.all_files = [rec]
    col.prepare(ctx)
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == rec.count('txt|persName', 'txt|placeName') + 3   # + sender, addressee, sent_from
    pl = [m for m in ms if m.kind == 'place' and m.zone == 'txt'][0]
    assert pl.auth['geonames'] == '2778436'
    s = [m for m in ms if m.role == 'sender'][0]
    assert s.surface == 'Maximilian I.' and s.doc_year == 1514
    row = s.to_row()
    assert row['tier'] == 'T1' and row['licence'] == 'CC-BY-3.0'
