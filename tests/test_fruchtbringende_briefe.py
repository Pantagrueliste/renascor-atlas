import pytest

from ratlas import common as C
from ratlas.collections import get

RAW = 'originals_2026-10-15/oct-binary/oct__fruchtbringende-gesellschaft-briefe-1617-1626/files/briefe/'
real = pytest.mark.skipif(not (C.raw_root() / RAW / '171224.xml').exists(), reason='raw data not available')

SYNTH = b'''<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="edoc_x_1617"><teiHeader><fileDesc><titleStmt>
<title>A an B</title><author><name type="person"><name>Klaus Conermann</name></name></author></titleStmt>
<publicationStmt><date>2011</date></publicationStmt></fileDesc></teiHeader>
<text><body><div type="title"><head><rs type="person" ref="#a_x">Herr A</rs> an <rs type="person" ref="#b_y">Herr an der B</rs></head></div>
<p>In <rs type="place" ref="#dessau">Dessau</rs> traf <rs type="person" ref="#a_x">er</rs> Apollo.</p></body></text></TEI>'''


def _ctx():
    col = get('fruchtbringende_briefe')
    return col, col.make_context()


def test_synthetic():
    col, ctx = _ctx()
    rec = C.FileRecord(path=RAW + '171224.xml', coll='x')
    root = C.parse_xml(SYNTH)
    ms, es = col.extractor.extract_root(root, rec, ctx)
    assert es == []
    roles = [(m.surface, m.role) for m in ms]
    assert ('Herr A', 'sender') in roles and ('Herr an der B', 'addressee') in roles
    assert not any(m.zone == 'hdr' for m in ms)                  # the editors in <author> are dropped
    assert all(m.doc_year == 1617 and m.date_source == 'filename' for m in ms)
    place = [m for m in ms if m.kind == 'place'][0]
    assert place.local_key == 'fruchtbringende_briefe:dessau' and place.in_register is False
    assert place.context and len(place.context) <= 120


@real
def test_real_letter():
    col, ctx = _ctx()
    rec = next(r for r in col.iter_files(ctx) if r.path.endswith('/171224.xml'))
    ms, _ = col.extract(rec, ctx)
    assert len(ms) == rec.count('txt|rs')
    s = [m for m in ms if m.role == 'sender']
    assert s and s[0].local_key == 'fruchtbringende_briefe:anhalt_dessau_johann_casimir'
    assert ms[0].doc_id == '171224' and ms[0].doc_year == 1617 and ms[0].in_period is True
    row = ms[0].to_row()
    assert row['auth']['other'] == []
